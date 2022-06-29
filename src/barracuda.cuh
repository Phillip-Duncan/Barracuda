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
#define MSTACK_SPECIALS 1
#endif

#ifndef MSTACK_LOOP_NEST_LIMIT
#define MSTACK_LOOP_NEST_LIMIT 24
#endif

#include <stdio.h>

#include "stack.cuh" // Include this before specials.cuh

#if MSTACK_SPECIALS==1
    #include "specials.cuh"
#endif

template<class I, class F, class L>
__device__
void eval(I type, I* stack, I &stackidx, I &stacksize, long long* opstack, I &opstackidx,
F* valuestack, I &valuestackidx, double* outputstack, I &outputstackidx, 
L tid, I nt, Vars<F> &variables, I* loop_stack, I &loop_idx )
{
    long long op = pop(opstack,opstackidx);
    F value = pop(valuestack, valuestackidx);

    // Is an ordinary operation
    if (type==0) {
        //op = pop(opstack,opstackidx);
        operation<F>(op, outputstack, outputstackidx, nt, 0, variables);
        return;
    }
    // Is a value
    else if (type==1) {
        //value = pop(valuestack, valuestackidx);
        push_t(outputstack, outputstackidx, value, nt);
        return;
    }
    // JUMP/Goto statement
    else if (type==2) {
        value = pop_t(outputstack, outputstackidx, nt);
        jmp(stack, stackidx, stacksize, opstackidx, valuestackidx, (I)value);
    }
    // JUMP IF == 0 / conditional goto statement
    else if (type==3) {
        value = pop_t(outputstack, outputstackidx, nt);
        I cond = (I)pop_t(outputstack, outputstackidx, nt);
        if (cond==0) {
            jmp(stack, stackidx, stacksize, opstackidx, valuestackidx, (I)value);
        }
        else {
        }
    }
    // Combinational instructions.
    else if (type==4) {
        push_t(outputstack, outputstackidx, value, nt);
        operation<F>(op, outputstack, outputstackidx, nt, 0, variables);
        return;
    }
    else if (type==5) {
        operation<F>(op, outputstack, outputstackidx, nt, 0, variables);
        push_t(outputstack, outputstackidx, value, nt);
        return;
    }
    // function pointer operation
    #if MSTACK_UNSAFE==1
        else if (type<0) {
            op = pop(opstack, opstackidx);
            operation<F>(type, op, outputstack, outputstackidx, nt, 0, variables);
            return;
        }
    #endif
    else if (type==99) {
        I imax = (I)pop_t(outputstack, outputstackidx, nt);
        I i = (I)pop_t(outputstack, outputstackidx, nt);
        // Store stack locations of loop entry, i and imax.
        loop_stack[5*loop_idx] = stackidx;
        loop_stack[5*loop_idx+1] = opstackidx;
        loop_stack[5*loop_idx+2] = valuestackidx;
        loop_stack[5*loop_idx+3] = i;
        loop_stack[5*loop_idx+4] = imax;
        loop_idx = loop_idx + 1;
        return;
    }
    else if (type==100) {
        // Compare values of i,imax to see whether loop has ended or continues
        I i = loop_stack[5*(loop_idx-1)+3];
        I imax = loop_stack[5*(loop_idx-1)+4];
        i = i + 1;
        loop_stack[5*(loop_idx-1)+3] = i;
        // If loop has reached end decrement loop_idx and return
        if (i>=imax){
            loop_idx = loop_idx - 1;
            return;
        }
        // Otherwise goto start of loop.
        else {
            // Reset program counter
            //variables.PC = variables.PC + stackidx - loop_stack[5*(loop_idx-1)];
            variables.PC = stacksize - loop_stack[5*(loop_idx-1)];

            // Swap indicies/sizes
            stackidx = loop_stack[5*(loop_idx-1)];
            //stacksize = loop_stack[5*(loop_idx-1)];
            opstackidx = loop_stack[5*(loop_idx-1) + 1];
            //opstacksize = loop_stack[5*(loop_idx-1) + 1];
            valuestackidx = loop_stack[5*(loop_idx-1) + 2];
            //valuestacksize = loop_stack[5*(loop_idx-1) + 2];
            return;
        }
    }
}

template<class I, class F, class L>
__device__
inline F evaluateStackExpr(I* stack, I stacksize, long long* opstack, I opstacksize,
F* valuestack, I valuestacksize, double* outputstack, I outputstacksize, L tid, I nt, Vars<F> &variables ) 
{

    // Make local versions of idxs for each thread
    I l_outputstackidx  = tid;

    I l_stacksize         = stacksize;
    I l_stackidx          = stacksize;
    I l_opstackidx        = opstacksize;
    I l_valuestackidx     = valuestacksize;

    I type;

    variables.TID = tid;

    // array + idx for nested loop "slots"
    // Change this in future to be implementation-allocated.
    I loop_stack[5*MSTACK_LOOP_NEST_LIMIT];
    I loop_idx = 0;

    // Set program counter back to 0
    variables.PC = 0;

    while (l_stackidx>0) {

        // "Pop type from stack"
        type = pop(stack,l_stackidx);

        eval(type, stack, l_stackidx, l_stacksize, opstack, l_opstackidx, valuestack, 
                l_valuestackidx, outputstack, l_outputstackidx,
                tid, nt, variables, loop_stack, loop_idx);

        // Advance program counter
        variables.PC = l_stacksize - l_stackidx;
    }
    // Return the final result from the outputstack
    return outputstack[tid];
}


// Function overload for when expression contains no Variables and struct not provided.
template<class I, class F, class L>
__device__
inline F evaluateStackExpr(I* stack, I stacksize, long long* opstack, I opstacksize,
F* valuestack, I valuestacksize, F* outputstack, I outputstacksize, L tid, I nt ) {
    Vars<F> Variables;
    return evaluateStackExpr(stack, stacksize, opstack, opstacksize,
        valuestack, valuestacksize, outputstack, outputstacksize, tid, nt, Variables);
}


#endif