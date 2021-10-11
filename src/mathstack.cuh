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

#ifndef MSTACK_LOOP_NEST_LIMIT
#define MSTACK_LOOP_NEST_LIMIT 24
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
L tid, I nt, Vars<F> &variables, I* loop_stack, I &loop_idx )
{
    LI op;
    F value;

    // Is an ordinary operation
    if (type==0) {
        op = pop(opstack,opstackidx,opstacksize);
        operation<F>(op, outputstack, outputstackidx, outputstacksize, nt, 0, variables);
        return;
    }
    // Is a value
    else if (type==1) {
        value = pop(valuestack,valuestackidx,valuestacksize);
        push_t(outputstack, outputstackidx, outputstacksize ,value, nt);
        return;
    }
    // Goto statement
    else if (type==2) {
        value = pop(outputstack,outputstackidx,outputstacksize);
        // Make sure goto is bounded between 0 and alloc(stack), otherwise just go to end
        value = value >= 0 ? value : 0;
        value = value <= (variables.PC + stackidx) ? value : variables.PC + stackidx;
        // Adjust stacksize and stackidx to "goto" value.
        stacksize = (I)(variables.PC + stackidx - value);
        stackidx  = (I)(variables.PC + stackidx - value);
        // Adjust program counter to value.
        variables.PC = value;
        return;
    }
    // function pointer operation
    #if MSTACK_UNSAFE==1
        else if (type<0) {
            op = pop(opstack,opstackidx,opstacksize);
            operation<F>(type, op, outputstack, outputstackidx, outputstacksize, nt, 0, variables);
            return;
        }
    #endif
    else if (type==99) {
        I imax = (I)pop_t(outputstack, outputstackidx, outputstacksize, nt);
        I i = (I)pop_t(outputstack, outputstackidx, outputstacksize, nt);
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
        if (i==imax){
            loop_idx = loop_idx - 1;
            return;
        }
        // Otherwise goto start of loop.
        else {
            // Reset program counter
            variables.PC = variables.PC + stackidx - loop_stack[5*(loop_idx-1)];

            // Swap indicies/sizes
            stackidx = loop_stack[5*(loop_idx-1)];
            stacksize = loop_stack[5*(loop_idx-1)];
            opstackidx = loop_stack[5*(loop_idx-1) + 1];
            opstacksize = loop_stack[5*(loop_idx-1) + 1];
            valuestackidx = loop_stack[5*(loop_idx-1) + 2];
            valuestacksize = loop_stack[5*(loop_idx-1) + 2];
            return;
        }
        
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

    variables.TID = tid;

    // array + idx for nested loop "slots"
    I loop_stack[5*MSTACK_LOOP_NEST_LIMIT];
    I loop_idx = 0;

    //for (int i=0;i<stacksize;i++) {
    while (l_stackidx>0) {

        // "Pop type from stack"
        type = pop(stack,l_stackidx,l_stacksize);

        // Advance program counter
        variables.PC += 1;
        eval(type, stack, l_stackidx, l_stacksize, opstack, l_opstackidx, l_opstacksize, valuestack, 
                l_valuestackidx, l_valuestacksize, outputstack, l_outputstackidx, l_outputstacksize,
                tid, nt, variables, loop_stack, loop_idx);

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