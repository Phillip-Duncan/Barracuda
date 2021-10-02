/**
 * @file mathstack.cuh
 * @author Phillip Duncan (phillip.duncan-gelder@pg.canterbury.ac.nz)
 * @brief Header-only stack-based equation solver 
 * @version 0.1
 * @date 2021-09-05
 * 
 * @copyright Copyright (c) 2021
 * 
 */


#ifndef _MATHSTACK_CUH
#define _MATHSTACK_CUH

#ifndef MSTACK_UNSAFE
#define MSTACK_UNSAFE 1
#endif

#ifndef MSTACK_SPECIALS
#define MSTACK_SPECIALS 1
#endif

#ifndef MSTACK_NESTED_LOOPS_ALLOWED
#define MSTACK_NESTED_LOOPS_ALLOWED 1
#endif

#ifndef MSTACK_MAX_RECURSION_DEPTH
#define MSTACK_MAX_RECURSION_DEPTH 11
#endif

#include <stdio.h>

#include "stack.cuh" // Include this before specials.cuh

#if MSTACK_SPECIALS==1
    #include "specials.cuh"
#endif

template<class I, class F, class LF, class L, class LI>
__device__
void eval(I type, I* stack, I &stackidx, I &stacksize, LI* opstack, I &opstackidx, I &opstacksize,
F* valuestack, I &valuestackidx, I &valuestacksize, LF* outputstack, I &outputstackidx, I &outputstacksize, 
L tid, I nt, Vars<F> &variables, I r_depth )
{
    LI op;
    F value;

    if (r_depth>MSTACK_MAX_RECURSION_DEPTH) {
        return;
    }

    // Is an ordinary operation
    if (type==0) {
        op = pop(opstack,opstackidx,opstacksize);
        operation<F>(op, outputstack, outputstackidx, outputstacksize, nt, 0, variables);
        return;
    }
    // Is a value
    if (type==1) {
        value = pop(valuestack,valuestackidx,valuestacksize);
        push_t(outputstack, outputstackidx, outputstacksize ,value, nt);
        return;
    }
    // function pointer operation
    if (type<0) {
        #if MSTACK_UNSAFE==1
            op = pop(opstack,opstackidx,opstacksize);
            operation<F>(type, op, outputstack, outputstackidx, outputstacksize, nt, 0, variables);
            return;
        #else
            return;
        #endif
    }

    if (type==2) {
        I i_idx_reset = 0;
        I o_idx_reset = 0;
        I v_idx_reset = 0;
        I imax;
        I i;
        imax = pop_t(outputstack, outputstackidx, outputstacksize, nt);
        i    = pop_t(outputstack, outputstackidx, outputstacksize, nt);
            
        for(;i<imax;i++){
            i_idx_reset = 0;
            o_idx_reset = 0;
            v_idx_reset = 0;
            
            while(true) {
                I t = pop(stack,stackidx,stacksize);
                i_idx_reset += 1;


                if (t==3)
                    break;

                if(t<=0)
                    o_idx_reset += 1;

                if(t==1)
                    v_idx_reset += 1;

                #if(MSTACK_NESTED_LOOPS_ALLOWED==1)
                    eval(t, stack, stackidx, stacksize, opstack, opstackidx, opstacksize,
                        valuestack, valuestackidx, valuestacksize, outputstack,
                        outputstackidx,outputstacksize, tid, nt, variables,r_depth+1);
                #else
                    // Is an ordinary operation
                    if (t==0) {
                            op = pop(opstack,opstackidx,opstacksize);
                            operation<F>(op, outputstack, outputstackidx, outputstacksize, nt, 0, variables);
                    }
                    // Is a value
                    if (t==1) {
                        value = pop(valuestack,valuestackidx,valuestacksize);
                        push_t(outputstack, outputstackidx, outputstacksize ,value, nt);
                    }
                    // function pointer operation
                    if (t<0) {
                        #if MSTACK_UNSAFE==1
                        op = pop(opstack,opstackidx,opstacksize);
                        operation<F>(type, op, outputstack, outputstackidx, outputstacksize, nt, 0, variables);
                        #else
                        #endif
                    }
                #endif
            }
            // Increment indexes of stacks to loop origin
            stackidx          += i_idx_reset;
            stacksize         += i_idx_reset;
            opstackidx        += o_idx_reset;
            opstacksize       += o_idx_reset;
            valuestackidx     += v_idx_reset;
            valuestacksize    += v_idx_reset;
        }
        // Set stack indexes to loop end at termination
        stackidx          -= i_idx_reset;
        stacksize         -= i_idx_reset;
        opstackidx        -= o_idx_reset;
        opstacksize       -= o_idx_reset;
        valuestackidx     -= v_idx_reset;
        valuestacksize    -= v_idx_reset;
        return;
    }
} 

template<class I, class F, class LF, class L, class LI>
__device__
inline F evaluateStackExpr(I* stack, I stacksize, LI* opstack, LI opstacksize,
F* valuestack, I valuestacksize, LF* outputstack, I outputstacksize, L tid, I nt, Vars<F> &variables ) 
{

    // Make local versions of stack sizes and idxs for each thread
    I l_outputstacksize = outputstacksize;
    I l_outputstackidx  = tid;

    I l_stacksize         = stacksize;
    I l_stackidx          = stacksize;
    I l_opstacksize       = opstacksize;
    I l_opstackidx        = opstacksize;
    I l_valuestacksize    = valuestacksize;
    I l_valuestackidx     = valuestacksize;

    I type;
    //LI op;
    //F value;
    

    for (int i=0;i<stacksize;i++) {

        // "Pop type from stack"
        type = pop(stack,l_stackidx,l_stacksize);
        
        eval(type, stack, l_stackidx, l_stacksize, opstack, l_opstackidx, l_opstacksize, valuestack, 
                l_valuestackidx, l_valuestacksize, outputstack, l_outputstackidx, l_outputstacksize,
                tid, nt, variables, 0);

    }
    // Return the final result from the outputstack
    return outputstack[tid];
}


// Function overload for when expression contains no Variables and struct not provided.
template<class I, class F, class L, class LI>
__device__
inline F evaluateStackExpr(I* stack, I stacksize, LI* opstack, LI opstacksize,
F* valuestack, I valuestacksize, F* outputstack, I outputstacksize, L tid, I nt ) {
    Vars<F> Variables;
    return evaluateStackExpr(stack, stacksize, opstack, opstacksize,
        valuestack, valuestacksize, outputstack, outputstacksize, tid, nt, Variables);
}


#endif