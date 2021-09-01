
#ifndef _MATHSTACK_CUH
#define _MATHSTACK_CUH

#ifndef MSTACK_UNSAFE
#define MSTACK_UNSAFE 1
#endif

#include <stdio.h>

enum OPCODES {

    // Null instruction
    OPNULL = 0x0,

    // Basic opcodes
    ADD = 0x3CC, SUB, MUL,
    DIV, AND, NAND, OR, NOR,
    XOR, NOT, INC, DEC, SWAP,


    // Mathematical operator opcodes 
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE
    ACOS = 0x798, ACOSH, ASIN, ASINH,
    ATAN, ATAN2, ATANH, CBRT, CEIL,
    CPYSGN, COS, COSH, COSPI, BESI0,
    BESI1, ERF, ERFC, ERFCI, ERFCX,
    ERFI, EXP, EXP10, EXP2, EXPM1,
    FABS, FDIM, FLOOR, FMA, FMAX,
    FMIN, FMOD, FREXP, HYPOT,
    ILOGB, ISFIN, ISINF, ISNAN,
    BESJ0, BESJ1, BESJN, LDEXP,
    LGAMMA, LLRINT, LLROUND, LOG,
    LOG10, LOG1P, LOG2, LOGB,
    LRINT, LROUND, MAX, MIN, MODF,
    NEARINT, NXTAFT, NORM, NORM3D,
    NORM4D, NORMCDF, NORMCDFINV,
    POW, RCBRT, REM, REMQUO,
    RHYPOT, RINT, RNORM, RNORM3D,
    RNORM4D, ROUND, RSQRT, SCALBLN,
    SCALBN, SGNBIT, SIN, SINH, SINPI,
    SQRT, TAN, TANH, TGAMMA, TRUNC,
    BESY0, BESY1, BESYN,

    // Load variable opcodes
    LDA = 0x12FC, LDB, LDC,
    LDD, LDE, LDF, LDG, LDH,
    LDI, LDJ, LDK, LDL, LDM,
    LDN, LDO, LDP, LDQ, LDR,
    LDS, LDT, LDU, LDV, LDW,
    LDX, LDY, LDZ,
    LDDT, LDDX, LDDY, LDDZ,

    // Receive/Store variable opcodes
    RCA = 0x16C8,
};

template<class F>
struct Vars {
    __device__ Vars(): a(0),b(0),c(0),
    d(0),e(0),f(0),g(0),h(0),
    i(0),j(0),k(0),l(0),m(0),
    n(0),o(0),p(0),q(0),r(0),
    s(0),t(0),u(0),v(0),w(0),
    x(0),y(0),z(0),
    dt(0),dx(0),dy(0),dz(0) { }

    F a;F b;F c;F d;F e;F f;
    F g;F h;F i;F j;F k;F l;
    F m;F n;F o;F p;F q;F r;
    F s;F t;F u;F v;F w;F x;
    F y;F z;
    
    F dt;F dx;F dy;F dz;
};

template<class U, class I>
__device__
U pop(U* stack, I &stackidx, I &stacksize) {
    if(stacksize>0) {
        stacksize = stacksize-1;
        stackidx = stackidx-1;
        U value = stack[stackidx];
        return value;
    }
    else {
        return stack[stackidx];
    }
}

template<class U, class I>
__device__
U pop_t(U* stack, I &stackidx, I &stacksize, I nt ) {
    if(stacksize>0) {
        stacksize = stacksize-1;
        stackidx = stackidx-nt;
        return stack[stackidx];
    }
    else {
        return stack[stackidx];
    }
}

template<class U, class I>
__device__
void push(U* stack, I &stackidx, I &stacksize, U value) {
    stack[stackidx] = value;
    stacksize = stacksize+1;
    stackidx = stackidx+1;
}

template<class U, class I>
__device__
void push_t(U* stack, I &stackidx, I &stacksize, U value, I nt) {
    stack[stackidx] = value;
    stacksize = stacksize+1;
    stackidx = stackidx+nt;
}


// Operation function, out-sources the switch statement out of main eval function.
template<class F, class I, class LI>
__device__
void operation(LI op, F* outputstack, I &o_stackidx, I &o_stacksize, I nt, Vars<F> &variables) {
    F value, v1, v2;
    switch(op) {
        // Null operation
        case OPNULL:
        {
            break;
        }



        // Basic Operations
        case ADD:
        {
            push_t(outputstack,o_stackidx,o_stacksize, pop_t(outputstack,o_stackidx,o_stacksize,nt)
            + pop_t(outputstack,o_stackidx,o_stacksize,nt), nt);
            break;
        }
        case SUB:
        {
            v1 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = v2 - v1;
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case MUL:
        {
            push_t(outputstack,o_stackidx,o_stacksize, pop_t(outputstack,o_stackidx,o_stacksize,nt)
            * pop_t(outputstack,o_stackidx,o_stacksize,nt), nt);
            break;
        }
        case DIV:
        {
            // Might need to handle divide by -,+ 0 later.. (currently just returns -inf,inf)
            v1 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = v2/v1;
            push_t(outputstack,o_stackidx,o_stacksize, value , nt);
            break;
        }
        case AND: 
        {
            push_t(outputstack,o_stackidx,o_stacksize, (F)((I)pop_t(outputstack,o_stackidx,o_stacksize,nt) & 
            (I)pop_t(outputstack,o_stackidx,o_stacksize,nt)), nt);
            break;
        }
        case NAND: 
        {
            push_t(outputstack,o_stackidx,o_stacksize, (F)!((I)pop_t(outputstack,o_stackidx,o_stacksize,nt) & 
            (I)pop_t(outputstack,o_stackidx,o_stacksize,nt)), nt);
            break;
        }
        case OR: 
        {
            push_t(outputstack,o_stackidx,o_stacksize, (F)((I)pop_t(outputstack,o_stackidx,o_stacksize,nt) | 
            (I)pop_t(outputstack,o_stackidx,o_stacksize,nt)), nt);
            break;
        }
        case NOR: 
        {
            push_t(outputstack,o_stackidx,o_stacksize, (F)!((I)pop_t(outputstack,o_stackidx,o_stacksize,nt) | 
            (I)pop_t(outputstack,o_stackidx,o_stacksize,nt)), nt);
            break;
        }
        case XOR: 
        {
            push_t(outputstack,o_stackidx,o_stacksize, (F)((I)pop_t(outputstack,o_stackidx,o_stacksize,nt) ^ 
            (I)pop_t(outputstack,o_stackidx,o_stacksize,nt)), nt);
            break;
        }
        case NOT: 
        {
            push_t(outputstack,o_stackidx,o_stacksize, (F)!((I)pop_t(outputstack,o_stackidx,o_stacksize,nt)), nt);
            break;
        }
        case INC:
        {
            value = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            push_t(outputstack,o_stackidx,o_stacksize, value++, nt);
            break;
        }
        case DEC:
        {
            value = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            push_t(outputstack,o_stackidx,o_stacksize, value--, nt);
            break;
        }
        case SWAP: 
        {
            v1 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            push_t(outputstack,o_stackidx,o_stacksize, v1 , nt);
            push_t(outputstack,o_stackidx,o_stacksize, v2 , nt);
            break;
        }



        // Mathematical Operators
        case ACOS:
        {
            value = acos(pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case ACOSH:
        {
            value = acosh(pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case ASIN:
        {
            value = asin(pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case ASINH:
        {
            value = asinh(pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case ATAN:
        {
            value = atan(pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case ATAN2:
        {
            v1 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = atan2(v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value , nt);
            break;
        }
        case ATANH:
        {
            value = atanh(pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case CBRT:
        {
            value = cbrt(pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case CEIL:
        {
            value = ceil(pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case SIN:
        {
            value = sin(pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }



        // Load Variable Operations
        case LDA:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.a, nt);
            break;
        }  
        case LDB:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.b, nt);
            break;
        }
    }
}



// Overloaded operation function for function pointers of arbitrary functions up to order 8. This is UNSAFE
#if MSTACK_UNSAFE==1
template<class F, class I, class LI>
__device__
void operation(I type, LI op, F* outputstack, I &o_stackidx, I &o_stacksize, I nt, Vars<F> &variables) {
    if (op==OPNULL)
        return;
    I nargs = abs(type);
    LI addr = op;
    F value, v1, v2, v3, v4, v5, v6, v7, v8;
    switch(nargs) {
        case 1:
        {
            v1 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            using fn = F(F);
            fn* func = (fn*) addr;
            value = (*func)(v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case 2:
        {
            v1 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            using fn = F(F,F);
            fn* func = (fn*) addr;
            value = (*func)(v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case 3:
        {
            v1 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v3 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            using fn = F(F,F,F);
            fn* func = (fn*) addr;
            value = (*func)(v3,v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case 4:
        {
            v1 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v3 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v4 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            using fn = F(F,F,F,F);
            fn* func = (fn*) addr;
            value = (*func)(v4,v3,v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case 5:
        {
            v1 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v3 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v4 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v5 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            using fn = F(F,F,F,F,F);
            fn* func = (fn*) addr;
            value = (*func)(v5,v4,v3,v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case 6:
        {
            v1 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v3 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v4 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v5 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v6 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            using fn = F(F,F,F,F,F,F);
            fn* func = (fn*) addr;
            value = (*func)(v6,v5,v4,v3,v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case 7:
        {
            v1 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v3 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v4 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v5 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v6 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v7 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            using fn = F(F,F,F,F,F,F,F);
            fn* func = (fn*) addr;
            value = (*func)(v7,v6,v5,v4,v3,v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case 8:
        {
            v1 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v3 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v4 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v5 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v6 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v7 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v8 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            using fn = F(F,F,F,F,F,F,F,F);
            fn* func = (fn*) addr;
            value = (*func)(v8,v7,v6,v5,v4,v3,v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
    }
}
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
            operation(op, outputstack, l_outputstackidx, l_outputstacksize, nt, variables);
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
            operation(type, op, outputstack, l_outputstackidx, l_outputstacksize, nt, variables);
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