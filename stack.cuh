/**
 * @file stack.cuh
 * @author Phillip Duncan (phillip.duncan-gelder@pg.canterbury.ac.nz)
 * @brief Separate header containing all the stack-based functions, operations, OPCODES to keep things in mathstack simpler.
 * @version 0.1
 * @date 2021-09-05
 * 
 * @copyright Copyright (c) 2021
 * 
 */

 #ifndef _STACK_CUH
 #define _STACK_CUH

//forward declaration
template<class I, class F, class LI>
inline __device__ F integrate(I intmethod, I maxstep, F accuracy, I functype, LI function, F llim, F ulim,
            F* outputstack, I &o_stackidx, I &o_stacksize, I nt);

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

    // Specials
    OPINTEGRATE = 0xF30,

    // Load variable opcodes
    LDA = 0x12FC, LDB, LDC,
    LDD, LDE, LDF, LDG, LDH,
    LDI, LDJ, LDK, LDL, LDM,
    LDN, LDO, LDP, LDQ, LDR,
    LDS, LDT, LDU, LDV, LDW,
    LDX, LDY, LDZ,
    LDDT, LDDX, LDDY, LDDZ,

    // Receive/Store variable opcodes
    RCA = 0x16C8, RCB, RCC,
    RCD, RCE, RCF, RCG, RCH,
    RCI, RCJ, RCK, RCL, RCM,
    RCN, RCO, RCP, RCQ, RCR,
    RCS, RCT, RCU, RCV, RCW,
    RCX, RCY, RCZ,
    RCDT, RCDX, RCDY, RCDZ,
};

template<class F>
struct Vars {
    __host__ __device__ Vars(): a(0),b(0),c(0),
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
inline U pop(U* stack, I &stackidx, I &stacksize) {
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
inline U pop_t(U* stack, I &stackidx, I &stacksize, I nt) {
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
inline void push(U* stack, I &stackidx, I &stacksize, U value) {
    stack[stackidx] = value;
    stacksize = stacksize+1;
    stackidx = stackidx+1;
}

template<class U, class I>
__device__
inline void push_t(U* stack, I &stackidx, I &stacksize, U value, I nt) {
    stack[stackidx] = value;
    stacksize = stacksize+1;
    stackidx = stackidx+nt;
}


/**
 * @brief Operation function, out-sources the switch statement out of main eval function.
 * 
 * @tparam I int/long type.
 * @tparam F float/double type.
 * @tparam LI address type.
 * @param op operation code (OPCODE).
 * @param outputstack pointer to the output stack.
 * @param o_stackidx output stack index.
 * @param o_stacksize output stack size.
 * @param nt total number of threads executing concurrently.
 * @param mode recursive/irrecursive mode (0 means recursion allowed, 1 prevents function recursion).
 * @param variables Struct symbolic variable storage for loading/storing values from symbols
 */
template<class F, class I, class LI>
__device__
inline void operation(LI op, F* outputstack, I &o_stackidx, I &o_stacksize, I nt, I mode, Vars<F> &variables) {
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
        case ERF:
        {
            value = erf(pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case SIN:
        {
            value = sin(pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case TAN:
        {
            value = tan(pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }



        // Special operations (if enabled)
        #if MSTACK_SPECIALS==1
            case OPINTEGRATE: 
            {   
                if(mode==0) {
                    I intmethod = (I) pop_t(outputstack, o_stackidx, o_stacksize, nt);
                    I maxsteps  = (I) pop_t(outputstack, o_stackidx, o_stacksize, nt);
                    F accuracy  = pop_t(outputstack, o_stackidx, o_stacksize, nt);
                    //I nofvars   = (I) pop_t(outputstack, o_stackidx, o_stacksize, nt); // Number of other variables function takes (implement later)
                    I functype  = (I) pop_t(outputstack, o_stackidx, o_stacksize, nt); // Whether to use in-built or user-defined function.
                    LI function = (LI) pop_t(outputstack, o_stackidx, o_stacksize, nt);
                    F ulim      = pop_t(outputstack, o_stackidx, o_stacksize, nt);
                    F llim      = pop_t(outputstack, o_stackidx, o_stacksize, nt);
                    value = integrate(intmethod,maxsteps,accuracy,functype,function,llim,ulim,
                                    outputstack, o_stackidx, o_stacksize, nt);
                    push_t(outputstack,o_stackidx,o_stacksize, value, nt);
                }
                break;
            }
        #endif



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
        case LDC:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.c, nt);
            break;
        }
        case LDD:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.d, nt);
            break;
        }
        case LDE:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.e, nt);
            break;
        }
        case LDF:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.f, nt);
            break;
        }
        case LDG:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.g, nt);
            break;
        }
        case LDH:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.h, nt);
            break;
        }
        case LDI:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.i, nt);
            break;
        }
        case LDJ:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.j, nt);
            break;
        }
        case LDK:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.k, nt);
            break;
        }
        case LDL:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.l, nt);
            break;
        }
        case LDM:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.m, nt);
            break;
        }
        case LDN:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.n, nt);
            break;
        }
        case LDO:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.o, nt);
            break;
        }
        case LDP:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.p, nt);
            break;
        }
        case LDQ:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.q, nt);
            break;
        }
        case LDR:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.r, nt);
            break;
        }
        case LDS:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.s, nt);
            break;
        }
        case LDT:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.t, nt);
            break;
        }
        case LDU:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.u, nt);
            break;
        }
        case LDV:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.v, nt);
            break;
        }
        case LDW:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.w, nt);
            break;
        }
        case LDX:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.x, nt);
            break;
        }
        case LDY:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.y, nt);
            break;
        }
        case LDZ:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.z, nt);
            break;
        }
        case LDDT:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.dt, nt);
            break;
        }
        case LDDX:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.dx, nt);
            break;
        }
        case LDDY:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.dy, nt);
            break;
        }
        case LDDZ:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.dz, nt);
            break;
        }



        // Receive/Store Variable Operations
        case RCA:
        {
            variables.a = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCB:
        {
            variables.b = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCC:
        {
            variables.c = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCD:
        {
            variables.d = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCE:
        {
            variables.e = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCF:
        {
            variables.f = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCG:
        {
            variables.g = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCH:
        {
            variables.h = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCI:
        {
            variables.i = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCJ:
        {
            variables.j = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCK:
        {
            variables.k = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCL:
        {
            variables.l = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCM:
        {
            variables.m = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCN:
        {
            variables.n = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCO:
        {
            variables.o = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCP:
        {
            variables.p = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCQ:
        {
            variables.q = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCR:
        {
            variables.r = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCS:
        {
            variables.s = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCT:
        {
            variables.t = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCU:
        {
            variables.u = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCV:
        {
            variables.v = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCW:
        {
            variables.w = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCX:
        {
            variables.x = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCY:
        {
            variables.y = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCZ:
        {
            variables.z = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCDT:
        {
            variables.dt = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCDX:
        {
            variables.dx = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCDY:
        {
            variables.dy = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCDZ:
        {
            variables.dz = pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
    }
}


// Overloaded normal operation function for no variables provided, just create an empty struct and pass in.
template<class F, class I, class LI>
__device__
inline void operation(LI op, F* outputstack, I &o_stackidx, I &o_stacksize, I nt, I mode) {
    Vars<F> variables;
    operation(op, outputstack, o_stackidx, o_stacksize, nt, mode, variables);
}


// Overloaded operation function for function pointers of arbitrary functions up to order 8. This is UNSAFE
#if MSTACK_UNSAFE==1
template<class F, class I, class LI>
__device__
inline void operation(I type, LI op, F* outputstack, I &o_stackidx, I &o_stacksize, I nt, I mode, Vars<F> &variables) {
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
// Overloaded function-pointer operation function for no variables provided, just create an empty struct and pass in.
template<class F, class I, class LI>
__device__
inline void operation(I type, LI op, F* outputstack, I &o_stackidx, I &o_stacksize, I nt, I mode) {
    Vars<F> variables;
    operation(type, op, outputstack, o_stackidx, o_stacksize, nt, mode, variables);
}
#endif
#endif