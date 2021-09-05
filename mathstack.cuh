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


#include <stdio.h>

//#if MSTACK_SPECIALS
//    #include "specials.cuh"
//#endif

#include "stack.cuh" // Include this before specials.cuh


#if MSTACK_SPECIALS==1
    #include "specials.cuh"
#endif

template<class I, class F, class L, class LI>
__device__
F evaluateStackExpr(I* stack, I stacksize, LI* opstack, LI opstacksize,
F* valuestack, I valuestacksize, F* outputstack, I outputstacksize, L tid, I nt, Vars<F> &variables ) 
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
    LI op;
    F value;
    

    for (int i=0;i<stacksize;i++) {

        // "Pop type from stack"
        type = pop(stack,l_stackidx,l_stacksize);
        
        // Is an ordinary operation
        if (type==0) {
            op = pop(opstack,l_opstackidx,l_opstacksize);
            operation(op, outputstack, l_outputstackidx, l_outputstacksize, nt, 0, variables);
        }
        // Is a value
        if (type==1) {
            value = pop(valuestack,l_valuestackidx,l_valuestacksize);
            push_t(outputstack, l_outputstackidx, l_outputstacksize ,value, nt);
        }
        // function pointer operation
        if (type<0) {
            #if MSTACK_UNSAFE==1
            op = pop(opstack,l_opstackidx,l_opstacksize);
            operation(type, op, outputstack, l_outputstackidx, l_outputstacksize, nt, 0, variables);
            #else
            #endif
        }
    }

    // Return the final result from the outputstack
    return outputstack[tid];
}


// Function overload for when expression contains no Variables and struct not provided.
template<class I, class F, class L, class LI>
__device__
F evaluateStackExpr(I* stack, I stacksize, LI* opstack, LI opstacksize,
F* valuestack, I valuestacksize, F* outputstack, I outputstacksize, L tid, I nt ) {
    Vars<F> Variables;
    return evaluateStackExpr(stack, stacksize, opstack, opstacksize,
        valuestack, valuestacksize, outputstack, outputstacksize, tid, nt, Variables);
}


#endif