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
template<class I, class F, class LF, class LI>
inline __device__ F integrate(I intmethod, I maxstep, F accuracy, I functype, LI function, F llim, F ulim,
            LF* outputstack, I &o_stackidx, I &o_stacksize, I nt);

enum OPCODES {

    // Null instruction
    OPNULL = 0x0,

    // Basic opcodes
    ADD = 0x3CC, SUB, MUL,
    DIV, AND, NAND, OR, NOR,
    XOR, NOT, INC, DEC, SWAP,
    DUP, OVER, LSHIFT, RSHIFT, 


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
    NNAN, NEARINT, NXTAFT, NORM, 
    NORM3D, NORM4D, NORMCDF, 
    NORMCDFINV, POW, RCBRT, REM,
    REMQUO, RHYPOT, RINT, RNORM,
    RNORM3D, RNORM4D, ROUND, RSQRT,
    SCALBLN, SCALBN, SGNBIT, SIN,
    SINH, SINPI, SQRT, TAN, TANH,
    TGAMMA, TRUNC, BESY0, BESY1,
    BESYN,

    // Specials
    OPINTEGRATE = 0xF30,

    // Load variable opcodes
    LDA = 0x12FC, LDB, LDC,
    LDD, LDE, LDF, LDG, LDH,
    LDI, LDJ, LDK, LDL, LDM,
    LDN, LDO, LDP, LDQ, LDR,
    LDS, LDT, LDU, LDV, LDW,
    LDX, LDY, LDZ,
    LDDX, LDDY, LDDZ,

    LDDT, LDA0, LDB0, LDC0,
    LDD0, LDE0, LDF0, LDG0, LDH0,
    LDI0, LDJ0, LDK0, LDL0, LDM0,
    LDN0, LDO0, LDP0, LDQ0, LDR0,
    LDS0, LDT0, LDU0, LDV0, LDW0,
    LDX0, LDY0, LDZ0,

    // Receive/Store variable opcodes
    RCA = 0x16C8, RCB, RCC,
    RCD, RCE, RCF, RCG, RCH,
    RCI, RCJ, RCK, RCL, RCM,
    RCN, RCO, RCP, RCQ, RCR,
    RCS, RCT, RCU, RCV, RCW,
    RCX, RCY, RCZ,
    RCDX, RCDY, RCDZ, RCDT
};

template<class F>
struct Vars {
    __host__ __device__ Vars(): 
    
    a(0),b(0),c(0),d(0),e(0),
    f(0),g(0),h(0),i(0),j(0),
    k(0),l(0),m(0),n(0),o(0),
    p(0),q(0),r(0),s(0),t(0),
    u(0),v(0),w(0),x(0),y(0),
    z(0),dx(0),dy(0),dz(0),
    
    dt(0),x0(0),y0(0),z0(0) { }

    // LD and RC capable variables
    F a,b,c,d,e,f,g,h,
    i,j,k,l,m,n,o,p,q,
    r,s,t,u,v,w,x,y,z,
    dx,dy,dz;

    // LD but NON-RC variables
    F dt,a0,b0,c0,d0,e0,
    f0,g0,h0,i0,j0,k0,l0,
    m0,n0,o0,p0,q0,r0,s0,
    t0,u0,v0,w0,x0,y0,z0;
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

// Overload push_t function where stack-value types differ so is thus broadcast. Useful for storing memory addresses, etc.
template<class T, class U, class I>
__device__
inline void push_t(T* stack, I &stackidx, I &stacksize, U value, I nt) {
    stack[stackidx] = (T)value;
    stacksize = stacksize+1;
    stackidx = stackidx+nt;
}


/**
 * @brief Operation function, out-sources the switch statement out of main eval function.
 * 
 * @tparam I int/long type.
 * @tparam F float/double type.
 * @tparam LF long float (double) for mixed STORAGE ONLY (addresses/values). Operations always cast to and done using (F) precision.
 * @tparam LI address type.
 * @param op operation code (OPCODE).
 * @param outputstack pointer to the output stack.
 * @param o_stackidx output stack index.
 * @param o_stacksize output stack size.
 * @param nt total number of threads executing concurrently.
 * @param mode recursive/irrecursive mode (0 means recursion allowed, 1 prevents function recursion).
 * @param variables Struct symbolic variable storage for loading/storing values from symbols
 */
template<class F, class LF, class I, class LI>
__device__
inline void operation(LI op, LF* outputstack, I &o_stackidx, I &o_stacksize, I nt, I mode, Vars<F> &variables) {
    F value, v1, v2;
    LF lvalue, lv1, lv2;
    switch(op) {
        // Null operation
        case OPNULL:
        {
            break;
        }



        // Basic Operations
        case ADD:
        {
            push_t(outputstack,o_stackidx,o_stacksize, (F)pop_t(outputstack,o_stackidx,o_stacksize,nt)
            + (F)pop_t(outputstack,o_stackidx,o_stacksize,nt), nt);
            break;
        }
        case SUB:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = v2 - v1;
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case MUL:
        {
            push_t(outputstack,o_stackidx,o_stacksize, (F)pop_t(outputstack,o_stackidx,o_stacksize,nt)
            * (F)pop_t(outputstack,o_stackidx,o_stacksize,nt), nt);
            break;
        }
        case DIV:
        {
            // Might need to handle divide by -,+ 0 later.. (currently just returns -inf,inf)
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = v2/v1;
            push_t(outputstack,o_stackidx,o_stacksize, value , nt);
            break;
        }
        case AND: 
        {
            push_t(outputstack,o_stackidx,o_stacksize, (LF)((LI)pop_t(outputstack,o_stackidx,o_stacksize,nt) & 
            (I)pop_t(outputstack,o_stackidx,o_stacksize,nt)), nt);
            break;
        }
        case NAND: 
        {
            push_t(outputstack,o_stackidx,o_stacksize, (LF)!((LI)pop_t(outputstack,o_stackidx,o_stacksize,nt) & 
            (I)pop_t(outputstack,o_stackidx,o_stacksize,nt)), nt);
            break;
        }
        case OR: 
        {
            push_t(outputstack,o_stackidx,o_stacksize, (LF)((LI)pop_t(outputstack,o_stackidx,o_stacksize,nt) | 
            (I)pop_t(outputstack,o_stackidx,o_stacksize,nt)), nt);
            break;
        }
        case NOR: 
        {
            push_t(outputstack,o_stackidx,o_stacksize, (LF)!((LI)pop_t(outputstack,o_stackidx,o_stacksize,nt) | 
            (I)pop_t(outputstack,o_stackidx,o_stacksize,nt)), nt);
            break;
        }
        case XOR: 
        {
            push_t(outputstack,o_stackidx,o_stacksize, (LF)((LI)pop_t(outputstack,o_stackidx,o_stacksize,nt) ^ 
            (I)pop_t(outputstack,o_stackidx,o_stacksize,nt)), nt);
            break;
        }
        case NOT: 
        {
            push_t(outputstack,o_stackidx,o_stacksize, (LF)!((LI)pop_t(outputstack,o_stackidx,o_stacksize,nt)), nt);
            break;
        }
        case INC:
        {
            value = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt)+1;
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case DEC:
        {
            value = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt)-1;
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case SWAP: 
        {
            lv1 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            lv2 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            push_t(outputstack,o_stackidx,o_stacksize, v1 , nt);
            push_t(outputstack,o_stackidx,o_stacksize, v2 , nt);
            break;
        }
        case DUP: 
        {
            lv1 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            push_t(outputstack,o_stackidx,o_stacksize, v1 , nt);
            push_t(outputstack,o_stackidx,o_stacksize, v1 , nt);
            break;
        }
        case OVER: 
        {
            lv1 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            lv2 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            push_t(outputstack,o_stackidx,o_stacksize, v2 , nt);
            push_t(outputstack,o_stackidx,o_stacksize, v1 , nt);
            push_t(outputstack,o_stackidx,o_stacksize, v2 , nt);
            break;
        }
        case LSHIFT:
        {
            lv1 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            lv2 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            lvalue = (LF)((LI)lv2 << (LI)lv1);
            push_t(outputstack,o_stackidx,o_stacksize, lvalue , nt);
        }
        case RSHIFT:
        {
            lv1 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            lv2 = pop_t(outputstack,o_stackidx,o_stacksize,nt);
            lvalue = (LF)((LI)lv2 >> (LI)lv1);
            push_t(outputstack,o_stackidx,o_stacksize, lvalue , nt);
        }



        // Mathematical Operators
        case ACOS:
        {
            value = acos((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case ACOSH:
        {
            value = acosh((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case ASIN:
        {
            value = asin((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case ASINH:
        {
            value = asinh((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case ATAN:
        {
            value = atan((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case ATAN2:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = atan2(v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value , nt);
            break;
        }
        case ATANH:
        {
            value = atanh((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case CBRT:
        {
            value = cbrt((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case CEIL:
        {
            value = ceil((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case CPYSGN:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = copysign(v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case COS:
        {
            value = cos((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case COSH:
        {
            value = cosh((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case COSPI:
        {
            value = cospi((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case BESI0:
        {
            value = cyl_bessel_i0((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case BESI1:
        {
            value = cyl_bessel_i1((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case ERF:
        {
            value = erf((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case ERFC:
        {
            value = erfc((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case ERFCI:
        {
            value = erfcinv((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case ERFCX:
        {
            value = erfcx((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case ERFI:
        {
            value = erfinv((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case EXP:
        {
            value = exp((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case EXP10:
        {
            value = exp10((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case EXP2:
        {
            value = exp2((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case EXPM1:
        {
            value = expm1((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case FABS:
        {
            value = fabs((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case FDIM:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = fdim(v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case FLOOR:
        {
            value = floor((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case FMA:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            F v3 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = fma(v3,v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case FMAX:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = fmax(v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case FMIN:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = fmin(v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case FMOD:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = fmod(v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case FREXP:
        {
            I* i_ptr = (I*)((LI)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = frexp(v2,i_ptr);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case HYPOT:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = hypot(v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case ILOGB:
        {
            value = ilogb((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case ISFIN:
        {
            value = isfinite((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case ISINF:
        {
            value = isinf((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case ISNAN:
        {
            value = isnan((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case BESJ0:
        {
            value = j0((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case BESJ1:
        {
            value = j1((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case BESJN:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (I)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = jn((I)v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case LDEXP:
        {
            v1 = (I)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = ldexp(v2,(I)v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case LGAMMA:
        {
            value = lgamma((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case LLRINT:
        {
            push_t(outputstack,o_stackidx,o_stacksize, llrint((F)pop_t(outputstack,o_stackidx,o_stacksize,nt)), nt);
            break;
        }
        case LLROUND:
        {
            push_t(outputstack,o_stackidx,o_stacksize, llround((F)pop_t(outputstack,o_stackidx,o_stacksize,nt)), nt);
            break;
        }
        case LOG:
        {
            value = log((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case LOG10:
        {
            value = log10((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case LOG1P:
        {
            value = log1p((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case LOG2:
        {
            value = log2((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case LOGB:
        {
            value = logb((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case LRINT:
        {
            push_t(outputstack,o_stackidx,o_stacksize, lrint((F)pop_t(outputstack,o_stackidx,o_stacksize,nt)), nt);
            break;
        }
        case LROUND:
        {
            push_t(outputstack,o_stackidx,o_stacksize, lround((F)pop_t(outputstack,o_stackidx,o_stacksize,nt)), nt);
            break;
        }
        case MAX:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = max(v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case MIN:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = min(v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case MODF:
        {
            F* f_ptr = (F*)((LI)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = modf(v2,f_ptr);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case NNAN:
        {
            char* char_ptr = (char*)((LI)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            value = nan(char_ptr);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case NEARINT:
        {
            value = nearbyint((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case NXTAFT:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = nextafter(v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case NORM:
        {
            double* d_ptr = (double*)((LI)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            v2 = (I)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = norm((I)v2,d_ptr);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case NORM3D:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            F v3 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = norm3d(v3,v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case NORM4D:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            F v3 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            F v4 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = norm4d(v4,v3,v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case NORMCDF:
        {
            value = normcdf((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case NORMCDFINV:
        {
            value = normcdfinv((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case POW:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = pow(v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case RCBRT:
        {
            value = rcbrt((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case REM:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = remainder(v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case REMQUO:
        {
            I* i_ptr = (I*)((LI)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = remquo(v2,v1,i_ptr);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case RHYPOT:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = rhypot(v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case RINT:
        {
            value = rint((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case RNORM:
        {
            double* d_ptr = (double*)((LI)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            v2 = (I)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = rnorm((I)v2,d_ptr);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case RNORM3D:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            F v3 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = rnorm3d(v3,v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case RNORM4D:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            F v3 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            F v4 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = rnorm4d(v4,v3,v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case ROUND:
        {
            value = round((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case RSQRT:
        {
            value = rsqrt((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case SCALBLN:
        {
            LI liv1 = (LI)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = scalbln(v2,liv1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case SCALBN:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = scalbn(v2,(I)v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case SGNBIT:
        {
            value = signbit((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case SIN:
        {
            value = sin((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case SINH:
        {
            value = sinh((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case SINPI:
        {
            value = sinpi((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case SQRT:
        {
            value = sqrt((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case TAN:
        {
            value = tan((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case TANH:
        {
            value = tanh((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case TGAMMA:
        {
            value = tgamma((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case TRUNC:
        {
            value = trunc((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case BESY0:
        {
            value = y0((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case BESY1:
        {
            value = y1((F)pop_t(outputstack,o_stackidx,o_stacksize,nt));
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case BESYN:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (I)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            value = yn((I)v2,v1);
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
                    F accuracy  = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
                    //I nofvars   = (I) pop_t(outputstack, o_stackidx, o_stacksize, nt); // Number of other variables function takes (implement later)
                    I functype  = (I) pop_t(outputstack, o_stackidx, o_stacksize, nt); // Whether to use in-built or user-defined function.
                    LI function = (LI) pop_t(outputstack, o_stackidx, o_stacksize, nt);
                    F ulim      = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
                    F llim      = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
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
        case LDDT:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.dt, nt);
            break;
        }
        case LDA0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.a0, nt);
            break;
        }  
        case LDB0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.b0, nt);
            break;
        }
        case LDC0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.c0, nt);
            break;
        }
        case LDD0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.d0, nt);
            break;
        }
        case LDE0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.e0, nt);
            break;
        }
        case LDF0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.f0, nt);
            break;
        }
        case LDG0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.g0, nt);
            break;
        }
        case LDH0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.h0, nt);
            break;
        }
        case LDI0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.i0, nt);
            break;
        }
        case LDJ0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.j0, nt);
            break;
        }
        case LDK0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.k0, nt);
            break;
        }
        case LDL0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.l0, nt);
            break;
        }
        case LDM0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.m0, nt);
            break;
        }
        case LDN0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.n0, nt);
            break;
        }
        case LDO0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.o0, nt);
            break;
        }
        case LDP0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.p0, nt);
            break;
        }
        case LDQ0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.q0, nt);
            break;
        }
        case LDR0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.r0, nt);
            break;
        }
        case LDS0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.s0, nt);
            break;
        }
        case LDT0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.t0, nt);
            break;
        }
        case LDU0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.u0, nt);
            break;
        }
        case LDV0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.v0, nt);
            break;
        }
        case LDW0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.w0, nt);
            break;
        }
        case LDX0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.x0, nt);
            break;
        }
        case LDY0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.y0, nt);
            break;
        }
        case LDZ0:
        {
            push_t(outputstack, o_stackidx, o_stacksize , variables.z0, nt);
            break;
        }



        // Receive/Store Variable Operations
        case RCA:
        {
            variables.a = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCB:
        {
            variables.b = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCC:
        {
            variables.c = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCD:
        {
            variables.d = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCE:
        {
            variables.e = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCF:
        {
            variables.f = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCG:
        {
            variables.g = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCH:
        {
            variables.h = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCI:
        {
            variables.i = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCJ:
        {
            variables.j = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCK:
        {
            variables.k = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCL:
        {
            variables.l = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCM:
        {
            variables.m = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCN:
        {
            variables.n = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCO:
        {
            variables.o = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCP:
        {
            variables.p = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCQ:
        {
            variables.q = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCR:
        {
            variables.r = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCS:
        {
            variables.s = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCT:
        {
            variables.t = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCU:
        {
            variables.u = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCV:
        {
            variables.v = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCW:
        {
            variables.w = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCX:
        {
            variables.x = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCY:
        {
            variables.y = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCZ:
        {
            variables.z = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCDX:
        {
            variables.dx = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCDY:
        {
            variables.dy = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        case RCDZ:
        {
            variables.dz = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
            break;
        }
        //case RCDT:
        //{
        //    variables.dt = (F)pop_t(outputstack, o_stackidx, o_stacksize, nt);
        //    break;
        //}
    }
}


// Overloaded normal operation function for no variables provided, just create an empty struct and pass in.
template<class F, class LF, class I, class LI>
__device__
inline void operation(LI op, LF* outputstack, I &o_stackidx, I &o_stacksize, I nt, I mode) {
    Vars<F> variables;
    operation(op, outputstack, o_stackidx, o_stacksize, nt, mode, variables);
}


// Overloaded operation function for function pointers of arbitrary functions up to order 8. This is UNSAFE
#if MSTACK_UNSAFE==1
template<class F, class LF, class I, class LI>
__device__
inline void operation(I type, LI op, LF* outputstack, I &o_stackidx, I &o_stacksize, I nt, I mode, Vars<F> &variables) {
    if (op==OPNULL)
        return;
    I nargs = abs(type);
    LI addr = op;
    F value, v1, v2, v3, v4, v5, v6, v7, v8;
    switch(nargs) {
        case 1:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            using fn = F(F);
            fn* func = (fn*) addr;
            value = (*func)(v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case 2:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            using fn = F(F,F);
            fn* func = (fn*) addr;
            value = (*func)(v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case 3:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v3 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            using fn = F(F,F,F);
            fn* func = (fn*) addr;
            value = (*func)(v3,v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case 4:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v3 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v4 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            using fn = F(F,F,F,F);
            fn* func = (fn*) addr;
            value = (*func)(v4,v3,v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case 5:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v3 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v4 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v5 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            using fn = F(F,F,F,F,F);
            fn* func = (fn*) addr;
            value = (*func)(v5,v4,v3,v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case 6:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v3 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v4 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v5 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v6 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            using fn = F(F,F,F,F,F,F);
            fn* func = (fn*) addr;
            value = (*func)(v6,v5,v4,v3,v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case 7:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v3 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v4 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v5 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v6 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v7 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            using fn = F(F,F,F,F,F,F,F);
            fn* func = (fn*) addr;
            value = (*func)(v7,v6,v5,v4,v3,v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
        case 8:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v3 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v4 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v5 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v6 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v7 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            v8 = (F)pop_t(outputstack,o_stackidx,o_stacksize,nt);
            using fn = F(F,F,F,F,F,F,F,F);
            fn* func = (fn*) addr;
            value = (*func)(v8,v7,v6,v5,v4,v3,v2,v1);
            push_t(outputstack,o_stackidx,o_stacksize, value, nt);
            break;
        }
    }
}
// Overloaded function-pointer operation function for no variables provided, just create an empty struct and pass in.
template<class F, class LF, class I, class LI>
__device__
inline void operation(I type, LI op, LF* outputstack, I &o_stackidx, I &o_stacksize, I nt, I mode) {
    Vars<F> variables;
    operation(type, op, outputstack, o_stackidx, o_stacksize, nt, mode, variables);
}
#endif
#endif