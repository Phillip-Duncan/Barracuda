/**
 * @file specials.cuh
 * @author Phillip Duncan (phillip.duncan-gelder@pg.canterbury.ac.nz)
 * @brief Implementation of some special functions not included in the standard CUDA library. 
 *        Note: These may be poorly optimized and likely have poor performance on GPU architectures.
 * @version 0.1
 * @date 2022-06-30
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef _SPECIALS_CUH
#define _SPECIALS_CUH

#ifndef MSTACK_SPECIALS_MMAXSTEP
#define MSTACK_SPECIALS_MMAXSTEP 10
#endif

#ifndef MSTACK_SPECIALS_DEBUG
#define MSTACK_SPECIALS_DEBUG 0
#endif

// Debugging
#if MSTACK_SPECIALS_DEBUG==1
    template<class I, class F>
    __device__ void dump_row(I i, F *R) {
    printf("R[%2zu] = ", i);
    for (I j = 0; j <= i; ++j){
        printf("%f ", R[j]);
    }
    printf("\n");
    }
#endif

/**
 * @brief Implementation of romberg integration technique (adapted from https://en.wikipedia.org/wiki/Romberg%27s_method).
 * 
 * @tparam I int/long type.
 * @tparam F float/double type.
 * @tparam long long address type.
 * @param max_steps max number of integration steps (this can be up to a global max of MSTACK_SPECIALS_MMAXSTEP).
 * @param acc desired accuracy.
 * @param functype function type (whether by native [library implented] or user-defined function pointer).
 * @param function either function opcode or function pointer depending on functype.
 * @param a lower integration limit.
 * @param b upper integration limit.
 * @param outputstack pointer to the output stack.
 * @param o_stackidx output stack index.
 * @param nt total number of threads executing concurrently.
 */
template<class I, class F>
__device__
inline F romberg(I max_steps, F acc, I functype, long long function, F a, F b,
    double* outputstack, I &o_stackidx, I nt) {
    Vars<F> dummy_variables;
    F R1[MSTACK_SPECIALS_MMAXSTEP], R2[MSTACK_SPECIALS_MMAXSTEP]; // buffers
    F *Rp = &R1[0], *Rc = &R2[0]; // Rp is previous row, Rc is current row
    F h = (b-a); //step size
    //Rp[0] = (f(a) + f(b))*h*.5; // first trapezoidal step
    switch(functype) {
        case 0:
        {
            push_t(outputstack,o_stackidx, a, nt);
            operation<F>(function, outputstack, o_stackidx, nt, 1, dummy_variables);
            push_t(outputstack,o_stackidx, b, nt);
            operation<F>(function, outputstack, o_stackidx, nt, 1, dummy_variables);
            Rp[0] = ( pop_t(outputstack,o_stackidx, nt) + pop_t(outputstack,o_stackidx, nt) )*h*0.5;
            break;
        }
        case 1: 
        {
            push_t(outputstack,o_stackidx, a, nt);
            operation<F>(-1,function, outputstack, o_stackidx,  nt, 1, dummy_variables);
            push_t(outputstack,o_stackidx, b, nt);
            operation<F>(-1,function, outputstack, o_stackidx,  nt, 1, dummy_variables);
            Rp[0] = ( pop_t(outputstack,o_stackidx, nt) + pop_t(outputstack,o_stackidx, nt) )*h*0.5;
            break;
        }
    }

    // Debugging
    #if MSTACK_SPECIALS_DEBUG==1 
        dump_row(0, Rp);
    #endif

    I mx_step = min(max_steps,MSTACK_SPECIALS_MMAXSTEP);

    for (I i = 1; i < mx_step; ++i) {
        h /= 2.;
        F c = 0;
        I ep = 1 << (i-1); //2^(n-1)
        for (I j = 1; j <= ep; ++j) {
            //c += f(a+(2*j-1)*h);
            switch(functype) {
                case 0:
                {
                    push_t(outputstack,o_stackidx, a+(2*j-1)*h, nt);
                    operation<F>(function, outputstack, o_stackidx, nt, 1, dummy_variables);
                    c += pop_t(outputstack,o_stackidx, nt);
                    break;
                }
                case 1:
                {
                    push_t(outputstack,o_stackidx, a+(2*j-1)*h, nt);
                    operation<F>(-1,function, outputstack, o_stackidx,  nt, 1, dummy_variables);
                    c += pop_t(outputstack,o_stackidx, nt);
                    break;
                }
            }
        }
        Rc[0] = h*c + .5*Rp[0]; //R(i,0)

        for (I j = 1; j <= i; ++j) {
            F n_k = pow(4, j);
            Rc[j] = (n_k*Rc[j-1] - Rp[j-1])/(n_k-1); // compute R(i,j)
        }

        // Debugging
        #if MSTACK_SPECIALS_DEBUG==1 
            dump_row(0, Rp);
        #endif

        if (i > 1 && fabs(Rp[i-1]-Rc[i]) < acc) {
            return Rc[i-1];
        }

        // swap Rn and Rc as we only need the last row
        F *rt = Rp;
        Rp = Rc;
        Rc = rt;
    }
    return Rp[mx_step-1]; // return our best guess
}


template<class I, class F>
__device__
inline F integrate(I intmethod, I maxstep, F accuracy, I functype, long long function, F llim, F ulim,
            double* outputstack, I &o_stackidx, I nt) {
    F value;
    switch(intmethod) {
        case 1: {
            value = romberg(maxstep, accuracy, functype, function, llim, ulim,
                outputstack, o_stackidx,  nt);
            break;
        }
    }

    return value;
}
#endif