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

#ifndef MSTACK_LOOP_NEST_LIMIT
#define MSTACK_LOOP_NEST_LIMIT 24
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

    // Built-in bounded for loop instructions.
    LOOP_ENTRY = 99,
    LOOP_EXIT

};

template<class F, class I, class L>
__device__
void eval(I type, I* stack, I &stackidx, I &stacksize, long long* opstack,
double* valuestack, double* outputstack, I &outputstackidx, 
L tid, I nt, double* userspace, I* loop_stack, I &loop_idx )
{
    long long op = read(opstack, stackidx);
    double value = read(valuestack, stackidx);

    switch(type) {
        case Instructions::OP: 
        {
            operation<F>(op, outputstack, outputstackidx, tid, nt, 0, userspace);
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
            operation<F>(op, outputstack, outputstackidx, tid, nt, 0, userspace);
            break;
        }
        case Instructions::SI_OP_VALUE: 
        {
            operation<F>(op, outputstack, outputstackidx, tid, nt, 0, userspace);
            push_t(outputstack, outputstackidx, value, nt);
            break;
        }
        case Instructions::LOOP_ENTRY: 
        {
            I imax = (I)__double_as_longlong(pop_t(outputstack, outputstackidx, nt));
            I i = (I)__double_as_longlong(pop_t(outputstack, outputstackidx, nt));
            // Store stack locations of loop entry, i and imax.
            loop_stack[3*loop_idx] = stackidx;
            loop_stack[3*loop_idx+1] = i;
            loop_stack[3*loop_idx+2] = imax;
            loop_idx = loop_idx + 1;
            break;
        }
        case Instructions::LOOP_EXIT:
        {
            // Compare values of i,imax to see whether loop has ended or continues
            I i = loop_stack[3*(loop_idx-1)+1];
            I imax = loop_stack[3*(loop_idx-1)+2];
            i = i + 1;
            loop_stack[3*(loop_idx-1)+1] = i;
            // If loop has reached end decrement loop_idx and return
            if (i>=imax){
                loop_idx = loop_idx - 1;
            }
            // Otherwise goto start of loop.
            else {
                // Swap indicies/sizes
                stackidx = loop_stack[3*(loop_idx-1)];
            }
            break;
        }
        default:
        {
            // function pointer operation
            #if MSTACK_UNSAFE==1
                if (type<0) {
                    operation<F>(type, op, outputstack, outputstackidx, nt);
                    break;
                }
            #endif
            break;
        }

    }
}

template<class F, class I, class L>
__device__
inline void evaluateStackExpr(I* stack, I stacksize, long long* opstack, double* valuestack,
    double* outputstack, I outputstacksize, L tid, I nt, double* userspace) 
{

    // Make local versions of idxs for each thread
    I l_outputstackidx  = tid;

    I l_stacksize         = stacksize;
    I l_stackidx          = stacksize;

    I type;

    // array + idx for nested loop "slots"
    // Change this in future to be implementation-allocated.
    I loop_stack[3*MSTACK_LOOP_NEST_LIMIT];
    I loop_idx = 0;

    while (l_stackidx>0) {

        // "Pop type from stack"
        type = pop(stack,l_stackidx);

        eval<F>(type, stack, l_stackidx, l_stacksize, opstack, valuestack, 
                outputstack, l_outputstackidx,
                tid, nt, userspace, loop_stack, loop_idx);

    }
}
#endif