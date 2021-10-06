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
L tid, I nt, Vars<F> &variables, I r_depth, I &i_reset, I &o_reset, I &v_reset )
{
    LI op;
    F value;
    if (r_depth>MSTACK_MAX_RECURSION_DEPTH) {
        return;
    }
    // Global (loop-recursive) resets for instruction, op, val stack index.
    i_reset = 0;
    o_reset = 0;
    v_reset = 0;
    
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
        I i_idx_reset = 0;
        I o_idx_reset = 0;
        I v_idx_reset = 0;
        I imax;
        I i;
        imax = (I)pop_t(outputstack, outputstackidx, outputstacksize, nt);
        i    = (I)pop_t(outputstack, outputstackidx, outputstacksize, nt);
        for(;i<imax;i++){
            i_idx_reset = 0;
            o_idx_reset = 0;
            v_idx_reset = 0;

            while(true) {
                I t = pop(stack,stackidx,stacksize);
                i_idx_reset += 1;


                if (t==100)
                {
                    break;
                }
                else if(t==0) {
                    o_idx_reset += 1;
                    op = pop(opstack,opstackidx,opstacksize);
                    operation<F>(op, outputstack, outputstackidx, outputstacksize, nt, 0, variables);
                    continue;
                }
                else if (t<0) {
                    #if MSTACK_UNSAFE==1
                    o_idx_reset += 1;
                    op = pop(opstack,opstackidx,opstacksize);
                    operation<F>(type, op, outputstack, outputstackidx, outputstacksize, nt, 0, variables);
                    #else
                    #endif
                    continue;
                }
                else if(t==1) {
                    value = pop(valuestack,valuestackidx,valuestacksize);
                    push_t(outputstack, outputstackidx, outputstacksize ,value, nt);
                    v_idx_reset += 1;
                    continue;
                }
                #if(MSTACK_NESTED_LOOPS_ALLOWED==1)
                    else if(t==99) {
                        eval(t, stack, stackidx, stacksize, opstack, opstackidx, opstacksize,
                            valuestack, valuestackidx, valuestacksize, outputstack,
                            outputstackidx,outputstacksize, tid, nt, variables,r_depth+1, i_reset, o_reset, v_reset);
                        // At end of each eval add recursion resets to super() reset total.
                        i_idx_reset += i_reset;
                        o_idx_reset += o_reset;
                        v_idx_reset += v_reset;
                        continue;
                    }
                #endif
            }

            // Increment indexes of stacks to loop origin.
            stackidx          += i_idx_reset;// + i_reset;
            stacksize         += i_idx_reset;// + i_reset;
            opstackidx        += o_idx_reset;// + o_reset;
            opstacksize       += o_idx_reset;// + o_reset;
            valuestackidx     += v_idx_reset;// + v_reset;
            valuestacksize    += v_idx_reset;// + v_reset;
        }
        // Set stack indexes to loop end at termination.
        stackidx          -= i_idx_reset;// + i_reset;
        stacksize         -= i_idx_reset;// + i_reset;
        opstackidx        -= o_idx_reset;// + o_reset;
        opstacksize       -= o_idx_reset;// + o_reset;
        valuestackidx     -= v_idx_reset;// + v_reset;
        valuestacksize    -= v_idx_reset;// + v_reset;

        // Add local idx reset to global idx reset.
        i_reset += i_idx_reset;
        o_reset += o_idx_reset;
        v_reset += v_idx_reset;

        return;
    }
} 

// Overloaded init eval function
template<class I, class F, class LF, class L, class LI>
__device__
void eval(I type, I* stack, I &stackidx, I &stacksize, LI* opstack, I &opstackidx, I &opstacksize,
F* valuestack, I &valuestackidx, I &valuestacksize, LF* outputstack, I &outputstackidx, I &outputstacksize, 
L tid, I nt, Vars<F> &variables, I r_depth)
{
    static I i=0,o=0,v=0;
    eval(type, stack, stackidx, stacksize, opstack, opstackidx, opstacksize,
                            valuestack, valuestackidx, valuestacksize, outputstack,
                            outputstackidx,outputstacksize, tid, nt, variables,r_depth,i,o,v);
    return;
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

    //for (int i=0;i<stacksize;i++) {
    while (l_stackidx>0) {

        // "Pop type from stack"
        type = pop(stack,l_stackidx,l_stacksize);

        // Advance program counter
        variables.PC += 1;
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