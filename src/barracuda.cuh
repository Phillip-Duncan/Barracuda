/**
 * @file barracuda.cuh
 * @author Phillip Duncan (phillip.duncan-gelder@pg.canterbury.ac.nz)
 * @brief Header-only pseudo-stack-based equation solver 
 * @version 0.1
 * @date 2022-06-30
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#ifndef _BARRACUDA_CUH
#define _BARRACUDA_CUH

#ifndef MSTACK_UNSAFE
#define MSTACK_UNSAFE 1
#endif

#ifndef MSTACK_SPECIALS
#define MSTACK_SPECIALS 0
#endif

#include <stdio.h>

#include "stack.cuh" // Include this before specials.cuh

#if MSTACK_SPECIALS==1
    #include "specials.cuh"
#endif

enum Instructions {
    OP = 0,
    VALUE,
    JUMP,
    JUMPIFZERO,

    // Combination instructions (Single instruction, multiple actions).
    SI_VALUE_OP,
    SI_OP_VALUE,
};

template<class F, class I, class L>
__device__
void eval(I type, I* stack, I &stackidx, I &stacksize, long long* opstack,
double* valuestack, double* outputstack, I &outputstackidx, 
L tid, I nt, double* userspace, long long* userspace_sizes)
{
    long long op = read(opstack, stackidx);
    double value = read(valuestack, stackidx);

    switch(type) {
        case Instructions::OP: 
        {
            operation<F>(op, outputstack, outputstackidx, tid, nt, 0, userspace, userspace_sizes);
            break;
        }
        case Instructions::VALUE:
        {
            push_t(outputstack, outputstackidx, value, nt);
            break;
        }
        case Instructions::JUMP: 
        {
            value = pop_t(outputstack, outputstackidx, nt);
            jmp(stack, stackidx, stacksize, (I)__double_as_longlong(value));
            break;
        }
        case Instructions::JUMPIFZERO:
        {
            value = pop_t(outputstack, outputstackidx, nt);
            I cond = pop_t(outputstack, outputstackidx, nt);
            if (cond==0) {
                jmp(stack, stackidx, stacksize, (I)__double_as_longlong(value));
            }
            else {
            }
            break;
        }
        case Instructions::SI_VALUE_OP:
        {
            push_t(outputstack, outputstackidx, value, nt);
            operation<F>(op, outputstack, outputstackidx, tid, nt, 0, userspace, userspace_sizes);
            break;
        }
        case Instructions::SI_OP_VALUE: 
        {
            operation<F>(op, outputstack, outputstackidx, tid, nt, 0, userspace, userspace_sizes);
            push_t(outputstack, outputstackidx, value, nt);
            break;
        }
        default:
        {
            break;
        }

    }
}

template<class F, class I, class L>
__device__
inline void evaluateStackExpr(I* stack, I stacksize, long long* opstack, double* valuestack,
    double* outputstack, L tid, I nt, double* userspace, long long* userspace_sizes) 
{

    // Make local versions of idxs for each thread
    I l_outputstackidx  = tid;

    I l_stacksize         = stacksize;
    I l_stackidx          = stacksize;

    I type;

    while (l_stackidx>0) {

        // "Pop type from stack"
        type = pop(stack,l_stackidx);

        eval<F>(type, stack, l_stackidx, l_stacksize, opstack, valuestack, 
                outputstack, l_outputstackidx,
                tid, nt, userspace, userspace_sizes);

    }
}
#endif