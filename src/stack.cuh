/**
 * @file stack.cuh
 * @author Phillip Duncan (phillip.duncan-gelder@pg.canterbury.ac.nz)
 * @brief Separate header containing all the stack-based functions, operations, OPCODES to keep things in barracuda simpler.
 * @version 0.1
 * @date 2022-06-30
 * 
 * @copyright Copyright (c) 2022
 * 
 */

 #ifndef _STACK_CUH
 #define _STACK_CUH

//forward declaration
template<class I, class F>
inline __device__ F integrate(I intmethod, I maxstep, F accuracy, I functype, long long function, F llim, F ulim,
            double* outputstack, I &o_stackidx, I nt);

enum OPCODES {

    // Null instruction
    OPNULL = 0x0,

    // Read and write from specific location on stack.
    SREAD = 0x1, SWRITE,

    // Basic opcodes
    ADD = 0x3CC, SUB, MUL,
    DIV, AND, NAND, OR, NOR,
    XOR, NOT, INC, DEC, SWAP,
    DUP, OVER, DROP, LSHIFT,
    RSHIFT,

    // Basic-extended opcodes
    MALLOC, FREE, MEMCPY, MEMSET,
    READ, WRITE, ADD_P, SUB_P,
    TERNARY,

    // Simplified Compare opcodes
    EQ, GT, GTEQ, LT, LTEQ, 

    // Extra Memory instruction codes (Enum follows from Ternary)
    READ_U8, READ_U16, READ_U32, READ_U64,
    READ_I8, READ_I16, READ_I32, READ_I64,
    READ_F8, READ_F16, READ_F32, READ_F64,
    READ_C8, READ_C16, READ_C32, READ_C64,
    READ_UC8, READ_UC16, READ_UC32, READ_UC64,

    WRITE_U8, WRITE_U16, WRITE_U32, WRITE_U64,
    WRITE_I8, WRITE_I16, WRITE_I32, WRITE_I64,
    WRITE_F8, WRITE_F16, WRITE_F32, WRITE_F64,
    WRITE_C8, WRITE_C16, WRITE_C32, WRITE_C64,
    WRITE_UC8, WRITE_UC16, WRITE_UC32, WRITE_UC64,

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

    // Sys calls
    PRINTC = 0xB64, PRINTCT,
    PRINTF, PRINTFT,

    // Specials
    OPINTEGRATE = 0xF30,

    // Load special variables
    LDPC = 0x12FC, LDTID,
    LDNXPTR,

    // Lower (0) and upper (1) ranges for Load and Store into Nth variate userspace.
    LDNX0 = 0xF4240,
    LDNX1 = 0x4C4B40,

    RCNX0 = 0x4C4B41,
    RCNX1 = 0x895441



};

template<class F>
struct __align__(64) Vars {
    __host__ __device__ Vars(): 
    /*
    a(0),b(0),c(0),d(0),e(0),
    f(0),g(0),h(0),i(0),j(0),
    k(0),l(0),m(0),n(0),o(0),
    p(0),q(0),r(0),s(0),t(0),
    u(0),v(0),w(0),x(0),y(0),
    z(0),dx(0),dy(0),dz(0),
    
    dt(0),a0(0),b0(0),c0(0),
    d0(0),e0(0),f0(0),g0(0),
    h0(0),i0(0),j0(0),k0(0),
    l0(0),m0(0),n0(0),o0(0),
    p0(0),q0(0),r0(0),s0(0),
    t0(0),u0(0),v0(0),w0(0),
    x0(0),y0(0),z0(0),*/

    PC(0), TID(0) { }

    // LD and RC capable variables
    /*
    F a,b,c,d,e,f,g,h,
    i,j,k,l,m,n,o,p,q,
    r,s,t,u,v,w,x,y,z,
    dx,dy,dz;

    // LD but NON-RC variables
    F dt,a0,b0,c0,d0,e0,
    f0,g0,h0,i0,j0,k0,l0,
    m0,n0,o0,p0,q0,r0,s0,
    t0,u0,v0,w0,x0,y0,z0; */

    // Internal use only, but can be loaded
    unsigned int PC, TID;

    // Allocated User-space
    F* userspace;
};

template<class U, class I>
__device__
inline U pop(U* stack, I &stackidx) {
    if(stackidx>0) {
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
inline U pop_t(U* stack, I &stackidx, I nt) {
    if((stackidx-nt)>=0) {
        stackidx = stackidx-nt;
        return stack[stackidx];
    }
    else {
        return stack[stackidx];
    }
}

template<class U, class I>
__device__
inline void push(U* stack, I &stackidx, U value) {
    stack[stackidx] = value;
    stackidx = stackidx+1;
}

template<class U, class I>
__device__
inline void push_t(U* stack, I &stackidx, U value, I nt) {
    stack[stackidx] = value;
    stackidx = stackidx+nt;
}

// Overload push_t function where stack-value types differ so is thus broadcast. Useful for storing memory addresses, etc.
template<class T, class U, class I>
__device__
inline void push_t(T* stack, I &stackidx, U value, I nt) {
    stack[stackidx] = (T)value;
    stackidx = stackidx+nt;
}

// Jump/GOTO position on instruction stack
template<class T, class I>
__device__
inline void jmp(T* stack, I &stackidx, I stacksize, I &opstackidx, I &valuestackidx, I pos) {
    // Make sure goto is bounded between 0 and alloc(stack), otherwise just go to end (or beginning if <0)
    pos = pos >= 0 ? pos : 0;
    pos = pos <= stacksize ? pos : stacksize;
    // Adjust stackidx to "goto" pos.
    stackidx  = (I)(stacksize - pos);
    opstackidx = stackidx;
    valuestackidx = stackidx;
    // Adjust program counter to pos.
    //PC = pos;
    return;
}


/**
 * @brief Operation function, out-sources the switch statement out of main eval function.
 * 
 * @tparam I int/long type.
 * @tparam F float/double type.
 * @tparam double for mixed STORAGE ONLY (addresses/values). Operations always cast to and done using (F) precision.
 * @tparam long long address type.
 * @param op operation code (OPCODE).
 * @param outputstack pointer to the output stack.
 * @param o_stackidx output stack index.
 * @param nt total number of threads executing concurrently.
 * @param mode recursive/irrecursive mode (0 means recursion allowed, 1 prevents function recursion).
 * @param variables Struct symbolic variable storage for loading/storing values from symbols
 */
template<class F, class I>
__device__
inline void operation(long long op, double* outputstack, I &o_stackidx, I nt, I mode, Vars<F> &variables) {
    F value, v1, v2;
    double lvalue, lv1, lv2;
    switch(op) {
        // Null operation
        case OPNULL:
        {
            break;
        }
        
        // Stack read and write instructions
        case SREAD:
        {
            push_t(outputstack,o_stackidx, 
            outputstack[__double_as_longlong(pop_t(outputstack,o_stackidx,nt))*nt + variables.TID],nt);
            break;
        }
        case SWRITE:
        {
            lv1 = pop_t(outputstack,o_stackidx,nt);
            lv2 = pop_t(outputstack,o_stackidx,nt);
            outputstack[__double_as_longlong(lv2)*nt + variables.TID] = lv1;
            break;
        }



        // Basic Operations
        case ADD:
        {
            push_t(outputstack,o_stackidx, (F)pop_t(outputstack,o_stackidx,nt)
            + (F)pop_t(outputstack,o_stackidx,nt), nt);
            break;
        }
        case SUB:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            value = v2 - v1;
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case MUL:
        {
            push_t(outputstack,o_stackidx, (F)pop_t(outputstack,o_stackidx,nt)
            * (F)pop_t(outputstack,o_stackidx,nt), nt);
            break;
        }
        case DIV:
        {
            // Might need to handle divide by -,+ 0 later.. (currently just returns -inf,inf)
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            value = v2/v1;
            push_t(outputstack,o_stackidx, value , nt);
            break;
        }
        case AND: 
        {
            push_t(outputstack,o_stackidx, (double)((unsigned long)pop_t(outputstack,o_stackidx,nt) & 
            (unsigned long)pop_t(outputstack,o_stackidx,nt)), nt);
            break;
        }
        case NAND: 
        {
            push_t(outputstack,o_stackidx, (double)!((unsigned long)pop_t(outputstack,o_stackidx,nt) & 
            (unsigned long)pop_t(outputstack,o_stackidx,nt)), nt);
            break;
        }
        case OR: 
        {
            push_t(outputstack,o_stackidx, (double)((unsigned long)pop_t(outputstack,o_stackidx,nt) | 
            (unsigned long)pop_t(outputstack,o_stackidx,nt)), nt);
            break;
        }
        case NOR: 
        {
            push_t(outputstack,o_stackidx, (double)!((unsigned long)pop_t(outputstack,o_stackidx,nt) | 
            (unsigned long)pop_t(outputstack,o_stackidx,nt)), nt);
            break;
        }
        case XOR: 
        {
            push_t(outputstack,o_stackidx, (double)((unsigned long)pop_t(outputstack,o_stackidx,nt) ^ 
            (unsigned long)pop_t(outputstack,o_stackidx,nt)), nt);
            break;
        }
        case NOT: 
        {
            push_t(outputstack,o_stackidx, (double)!((unsigned long)pop_t(outputstack,o_stackidx,nt)), nt);
            break;
        }
        case INC:
        {
            value = (F)pop_t(outputstack,o_stackidx,nt)+1;
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case DEC:
        {
            value = (F)pop_t(outputstack,o_stackidx,nt)-1;
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case SWAP: 
        {
            lv1 = pop_t(outputstack,o_stackidx,nt);
            lv2 = pop_t(outputstack,o_stackidx,nt);
            push_t(outputstack,o_stackidx, lv1 , nt);
            push_t(outputstack,o_stackidx, lv2 , nt);
            break;
        }
        case DUP: 
        {
            lv1 = pop_t(outputstack,o_stackidx,nt);
            push_t(outputstack,o_stackidx, lv1 , nt);
            push_t(outputstack,o_stackidx, lv1 , nt);
            break;
        }
        case OVER: 
        {
            lv1 = pop_t(outputstack,o_stackidx,nt);
            lv2 = pop_t(outputstack,o_stackidx,nt);
            push_t(outputstack,o_stackidx, lv2 , nt);
            push_t(outputstack,o_stackidx, lv1 , nt);
            push_t(outputstack,o_stackidx, lv2 , nt);
            break;
        }
        case DROP:
        {
            pop_t(outputstack,o_stackidx,nt);
            break;
        }
        case LSHIFT:
        {
            lv1 = pop_t(outputstack,o_stackidx,nt);
            lv2 = pop_t(outputstack,o_stackidx,nt);
            lvalue = (double)((unsigned long)lv2 << (unsigned long)lv1);
            push_t(outputstack,o_stackidx, lvalue , nt);
            break;
        }
        case RSHIFT:
        {
            lv1 = pop_t(outputstack,o_stackidx,nt);
            lv2 = pop_t(outputstack,o_stackidx,nt);
            lvalue = (double)((unsigned long)lv2 >> (unsigned long)lv1);
            push_t(outputstack,o_stackidx, lvalue , nt);
            break;
        }



        // Memory/Pointers
        case MALLOC:
        {
            F* addr = (F*)malloc(__double_as_longlong(pop_t(outputstack,o_stackidx,nt)));
            push_t(outputstack,o_stackidx, __longlong_as_double((long long)addr) , nt);
            break;
        }
        case FREE:
        {
            F* addr = (F*)__double_as_longlong(pop_t(outputstack,o_stackidx,nt));
            if(addr!=NULL)
                free(addr);
            break;
        }
        case MEMCPY:
        {
            long long size     = __double_as_longlong(pop_t(outputstack,o_stackidx,nt));
            F* src      = (F*)__double_as_longlong(pop_t(outputstack,o_stackidx,nt));
            F* dest     = (F*)__double_as_longlong(pop_t(outputstack,o_stackidx,nt));
            memcpy(dest,src,size);
            break;
        }
        case MEMSET:
        {
            long long size     = __double_as_longlong(pop_t(outputstack,o_stackidx,nt));
            F val       = (F)pop_t(outputstack,o_stackidx,nt);
            F* src      = (F*)__double_as_longlong(pop_t(outputstack,o_stackidx,nt));
            memset(src,val,size);
            push_t(outputstack,o_stackidx,__longlong_as_double((long long)src), nt);
            break;
        }
        case READ:
        {
            F* addr = (F*)__double_as_longlong(pop_t(outputstack,o_stackidx,nt));
            if(addr!=NULL) {
                value = *addr;
                push_t(outputstack,o_stackidx,value, nt);
            }
            break;
        }
        case WRITE:
        {
            value = (F)pop_t(outputstack,o_stackidx,nt);
            F* addr = (F*)__double_as_longlong(pop_t(outputstack,o_stackidx,nt));
            if(addr!=NULL)
                *addr = value;
            break;
        }
        case ADD_P: 
        {   
            long long li1 = __double_as_longlong(pop_t(outputstack,o_stackidx,nt));
            long long addr = __double_as_longlong(pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, __longlong_as_double(addr + li1), nt);
            break;
        }
        case SUB_P: 
        {
            long long li1 = __double_as_longlong(pop_t(outputstack,o_stackidx,nt));
            long long addr = __double_as_longlong(pop_t(outputstack,o_stackidx,nt));
            long long val = addr - li1; //li2 - li1;
            push_t(outputstack,o_stackidx, __longlong_as_double(val), nt);
            break;
        }
        case TERNARY: 
        {
            lv1 = pop_t(outputstack,o_stackidx,nt);
            lv2 = pop_t(outputstack,o_stackidx,nt);
            double lv3 = pop_t(outputstack,o_stackidx,nt);
            lvalue = (lv3>0) ? lv2:lv1;
            push_t(outputstack,o_stackidx, lvalue , nt);
            break;
        }


        // Compare Operators
        case EQ: 
        {
            lv1 = pop_t(outputstack,o_stackidx,nt);
            lv2 = pop_t(outputstack,o_stackidx,nt);
            lvalue = lv1 == lv2;
            push_t(outputstack,o_stackidx,lvalue,nt);
            break;
        }
        case GT: 
        {
            lv1 = pop_t(outputstack,o_stackidx,nt);
            lv2 = pop_t(outputstack,o_stackidx,nt);
            lvalue = lv1 < lv2;
            push_t(outputstack,o_stackidx,lvalue,nt);
            break;
        }
        case GTEQ: 
        {
            lv1 = pop_t(outputstack,o_stackidx,nt);
            lv2 = pop_t(outputstack,o_stackidx,nt);
            lvalue = lv1 <= lv2;
            push_t(outputstack,o_stackidx,lvalue,nt);
            break;
        }
        case LT: 
        {
            lv1 = pop_t(outputstack,o_stackidx,nt);
            lv2 = pop_t(outputstack,o_stackidx,nt);
            lvalue = lv1 > lv2;
            push_t(outputstack,o_stackidx,lvalue,nt);
            break;
        }
        case LTEQ: 
        {
            lv1 = pop_t(outputstack,o_stackidx,nt);
            lv2 = pop_t(outputstack,o_stackidx,nt);
            lvalue = lv1 >= lv2;
            push_t(outputstack,o_stackidx,lvalue,nt);
            break;
        }



        // Mathematical Operators
        case ACOS:
        {
            value = acos((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case ACOSH:
        {
            value = acosh((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case ASIN:
        {
            value = asin((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case ASINH:
        {
            value = asinh((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case ATAN:
        {
            value = atan((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case ATAN2:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            value = atan2(v2,v1);
            push_t(outputstack,o_stackidx, value , nt);
            break;
        }
        case ATANH:
        {
            value = atanh((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case CBRT:
        {
            value = cbrt((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case CEIL:
        {
            value = ceil((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case CPYSGN:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            value = copysign(v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case COS:
        {
            value = cos((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case COSH:
        {
            value = cosh((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case COSPI:
        {
            value = cospi((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case BESI0:
        {
            value = cyl_bessel_i0((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case BESI1:
        {
            value = cyl_bessel_i1((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case ERF:
        {
            value = erf((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case ERFC:
        {
            value = erfc((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case ERFCI:
        {
            value = erfcinv((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case ERFCX:
        {
            value = erfcx((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case ERFI:
        {
            value = erfinv((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case EXP:
        {
            value = exp((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case EXP10:
        {
            value = exp10((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case EXP2:
        {
            value = exp2((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case EXPM1:
        {
            value = expm1((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case FABS:
        {
            value = fabs((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case FDIM:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            value = fdim(v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case FLOOR:
        {
            value = floor((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case FMA:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            F v3 = (F)pop_t(outputstack,o_stackidx,nt);
            value = fma(v3,v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case FMAX:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            value = fmax(v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case FMIN:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            value = fmin(v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case FMOD:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            value = fmod(v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case FREXP:
        {
            I* i_ptr = (I*)((long long)pop_t(outputstack,o_stackidx,nt));
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            value = frexp(v2,i_ptr);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case HYPOT:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            value = hypot(v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case ILOGB:
        {
            value = ilogb((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case ISFIN:
        {
            value = isfinite((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case ISINF:
        {
            value = isinf((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case ISNAN:
        {
            value = isnan((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case BESJ0:
        {
            value = j0((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case BESJ1:
        {
            value = j1((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case BESJN:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (I)pop_t(outputstack,o_stackidx,nt);
            value = jn((I)v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case LDEXP:
        {
            v1 = (I)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            value = ldexp(v2,(I)v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case LGAMMA:
        {
            value = lgamma((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case LLRINT:
        {
            push_t(outputstack,o_stackidx, llrint((F)pop_t(outputstack,o_stackidx,nt)), nt);
            break;
        }
        case LLROUND:
        {
            push_t(outputstack,o_stackidx, llround((F)pop_t(outputstack,o_stackidx,nt)), nt);
            break;
        }
        case LOG:
        {
            value = log((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case LOG10:
        {
            value = log10((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case LOG1P:
        {
            value = log1p((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case LOG2:
        {
            value = log2((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case LOGB:
        {
            value = logb((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case LRINT:
        {
            push_t(outputstack,o_stackidx, lrint((F)pop_t(outputstack,o_stackidx,nt)), nt);
            break;
        }
        case LROUND:
        {
            push_t(outputstack,o_stackidx, lround((F)pop_t(outputstack,o_stackidx,nt)), nt);
            break;
        }
        case MAX:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            value = max(v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case MIN:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            value = min(v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case MODF:
        {
            F* f_ptr = (F*)((long long)pop_t(outputstack,o_stackidx,nt));
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            value = modf(v2,f_ptr);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case NNAN:
        {
            char* char_ptr = (char*)((long long)pop_t(outputstack,o_stackidx,nt));
            value = nan(char_ptr);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case NEARINT:
        {
            value = nearbyint((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case NXTAFT:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            value = nextafter(v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case NORM:
        {
            double* d_ptr = (double*)((long long)pop_t(outputstack,o_stackidx,nt));
            v2 = (I)pop_t(outputstack,o_stackidx,nt);
            value = norm((I)v2,d_ptr);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case NORM3D:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            F v3 = (F)pop_t(outputstack,o_stackidx,nt);
            value = norm3d(v3,v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case NORM4D:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            F v3 = (F)pop_t(outputstack,o_stackidx,nt);
            F v4 = (F)pop_t(outputstack,o_stackidx,nt);
            value = norm4d(v4,v3,v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case NORMCDF:
        {
            value = normcdf((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case NORMCDFINV:
        {
            value = normcdfinv((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case POW:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            value = pow(v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case RCBRT:
        {
            value = rcbrt((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case REM:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            value = remainder(v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case REMQUO:
        {
            I* i_ptr = (I*)((long long)pop_t(outputstack,o_stackidx,nt));
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            value = remquo(v2,v1,i_ptr);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case RHYPOT:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            value = rhypot(v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case RINT:
        {
            value = rint((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case RNORM:
        {
            double* d_ptr = (double*)((long long)pop_t(outputstack,o_stackidx,nt));
            v2 = (I)pop_t(outputstack,o_stackidx,nt);
            value = rnorm((I)v2,d_ptr);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case RNORM3D:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            F v3 = (F)pop_t(outputstack,o_stackidx,nt);
            value = rnorm3d(v3,v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case RNORM4D:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            F v3 = (F)pop_t(outputstack,o_stackidx,nt);
            F v4 = (F)pop_t(outputstack,o_stackidx,nt);
            value = rnorm4d(v4,v3,v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case ROUND:
        {
            value = round((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case RSQRT:
        {
            value = rsqrt((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case SCALBLN:
        {
            long long liv1 = (long long)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            value = scalbln(v2,liv1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case SCALBN:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            value = scalbn((double)v2,(I)v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case SGNBIT:
        {
            value = signbit((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case SIN:
        {
            value = sin((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case SINH:
        {
            value = sinh((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case SINPI:
        {
            value = sinpi((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case SQRT:
        {
            value = sqrt((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case TAN:
        {
            value = tan((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case TANH:
        {
            value = tanh((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case TGAMMA:
        {
            value = tgamma((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case TRUNC:
        {
            value = trunc((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case BESY0:
        {
            value = y0((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case BESY1:
        {
            value = y1((F)pop_t(outputstack,o_stackidx,nt));
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case BESYN:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (I)pop_t(outputstack,o_stackidx,nt);
            value = yn((I)v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }



        //Syscall operations
        case PRINTC:
        {
            char char_v = (char)(long long)pop_t(outputstack, o_stackidx, nt);
            printf("%c",char_v);
            break;
        }
        case PRINTCT:
        {
            char char_v = (char)(long long)pop_t(outputstack, o_stackidx, nt);
            long long thread = (long long)pop_t(outputstack, o_stackidx, nt);
            if (thread==variables.TID)
                printf("%c",char_v);
            break;
        }
        case PRINTF:
        {
            value = (F)(long long)pop_t(outputstack, o_stackidx, nt);
            printf("%f",value);
            break;
        }
        case PRINTFT:
        {
            value = (F)(long long)pop_t(outputstack, o_stackidx, nt);
            long long thread = (long long)pop_t(outputstack, o_stackidx, nt);
            if (thread==variables.TID)
                printf("%f",value);
            break;
        }



        // Special operations (if enabled)
        #if MSTACK_SPECIALS==1
            case OPINTEGRATE: 
            {   
                if(mode==0) {
                    I intmethod = (I) pop_t(outputstack, o_stackidx, nt);
                    I maxsteps  = (I) pop_t(outputstack, o_stackidx, nt);
                    F accuracy  = (F)pop_t(outputstack, o_stackidx, nt);
                    //I nofvars   = (I) pop_t(outputstack, o_stackidx, nt); // Number of other variables function takes (implement later)
                    I functype  = (I) pop_t(outputstack, o_stackidx, nt); // Whether to use in-built or user-defined function.
                    long long function = (long long) pop_t(outputstack, o_stackidx, nt);
                    F ulim      = (F)pop_t(outputstack, o_stackidx, nt);
                    F llim      = (F)pop_t(outputstack, o_stackidx, nt);
                    value = integrate(intmethod,maxsteps,accuracy,functype,function,llim,ulim,
                                    outputstack, o_stackidx, nt);
                    push_t(outputstack,o_stackidx, value, nt);
                }
                break;
            }
        #endif

        case LDPC:
        {
            push_t(outputstack, o_stackidx, variables.PC, nt);
            break;
        }
        case LDTID:
        {
            push_t(outputstack, o_stackidx, variables.TID, nt);
            break;
        }
        case LDNXPTR:
        {
            push_t(outputstack, o_stackidx, __longlong_as_double(
                (long long)&variables.userspace[variables.TID + __double_as_longlong(pop_t(outputstack, o_stackidx, nt))*nt]), nt);
            break;
        }

        // Default case for Load/store vars from userspace, this ensures switch statement still a jump table.
        default:
        {
            // Load operation, push value to the stack from userspace indexed by tid and (op-LDNX0) delta.
            if (op >= LDNX0 && op <= LDNX1) {
                push_t(outputstack, o_stackidx, (double)variables.userspace[variables.TID + (op-LDNX0)*nt], nt);
            }
            // Store operation, store value from the stack to userspace indexed by tid and (op-RCNX0) delta.
            else if (op >= RCNX0 && op < RCNX1) {
                variables.userspace[variables.TID + (op-RCNX0)*nt] = (F)pop_t(outputstack, o_stackidx, nt);
            }
            // Else do nothing.
            else {
                break;
            }
            break;
        }
    }

}


// Overloaded normal operation function for no variables provided, just create an empty struct and pass in.
template<class F, class I>
__device__
inline void operation(long long op, double* outputstack, I &o_stackidx, I nt, I mode) {
    Vars<F> variables;
    operation(op, outputstack, o_stackidx, nt, mode, variables);
}


// Overloaded operation function for function pointers of arbitrary functions up to order 8. This is UNSAFE
#if MSTACK_UNSAFE==1
template<class F, class I>
__device__
inline void operation(I type, long long op, double* outputstack, I &o_stackidx, I nt, I mode, Vars<F> &variables) {
    if (op==OPNULL)
        return;
    I nargs = abs(type);
    long long addr = op;
    F value, v1, v2, v3, v4, v5, v6, v7, v8;
    switch(nargs) {
        case 1:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            using fn = F(F);
            fn* func = (fn*) addr;
            value = (*func)(v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case 2:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            using fn = F(F,F);
            fn* func = (fn*) addr;
            value = (*func)(v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case 3:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            v3 = (F)pop_t(outputstack,o_stackidx,nt);
            using fn = F(F,F,F);
            fn* func = (fn*) addr;
            value = (*func)(v3,v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case 4:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            v3 = (F)pop_t(outputstack,o_stackidx,nt);
            v4 = (F)pop_t(outputstack,o_stackidx,nt);
            using fn = F(F,F,F,F);
            fn* func = (fn*) addr;
            value = (*func)(v4,v3,v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case 5:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            v3 = (F)pop_t(outputstack,o_stackidx,nt);
            v4 = (F)pop_t(outputstack,o_stackidx,nt);
            v5 = (F)pop_t(outputstack,o_stackidx,nt);
            using fn = F(F,F,F,F,F);
            fn* func = (fn*) addr;
            value = (*func)(v5,v4,v3,v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case 6:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            v3 = (F)pop_t(outputstack,o_stackidx,nt);
            v4 = (F)pop_t(outputstack,o_stackidx,nt);
            v5 = (F)pop_t(outputstack,o_stackidx,nt);
            v6 = (F)pop_t(outputstack,o_stackidx,nt);
            using fn = F(F,F,F,F,F,F);
            fn* func = (fn*) addr;
            value = (*func)(v6,v5,v4,v3,v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case 7:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            v3 = (F)pop_t(outputstack,o_stackidx,nt);
            v4 = (F)pop_t(outputstack,o_stackidx,nt);
            v5 = (F)pop_t(outputstack,o_stackidx,nt);
            v6 = (F)pop_t(outputstack,o_stackidx,nt);
            v7 = (F)pop_t(outputstack,o_stackidx,nt);
            using fn = F(F,F,F,F,F,F,F);
            fn* func = (fn*) addr;
            value = (*func)(v7,v6,v5,v4,v3,v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
        case 8:
        {
            v1 = (F)pop_t(outputstack,o_stackidx,nt);
            v2 = (F)pop_t(outputstack,o_stackidx,nt);
            v3 = (F)pop_t(outputstack,o_stackidx,nt);
            v4 = (F)pop_t(outputstack,o_stackidx,nt);
            v5 = (F)pop_t(outputstack,o_stackidx,nt);
            v6 = (F)pop_t(outputstack,o_stackidx,nt);
            v7 = (F)pop_t(outputstack,o_stackidx,nt);
            v8 = (F)pop_t(outputstack,o_stackidx,nt);
            using fn = F(F,F,F,F,F,F,F,F);
            fn* func = (fn*) addr;
            value = (*func)(v8,v7,v6,v5,v4,v3,v2,v1);
            push_t(outputstack,o_stackidx, value, nt);
            break;
        }
    }
}
// Overloaded function-pointer operation function for no variables provided, just create an empty struct and pass in.
template<class F, class I>
__device__
inline void operation(I type, long long op, double* outputstack, I &o_stackidx, I nt, I mode) {
    Vars<F> variables;
    operation(type, op, outputstack, o_stackidx, nt, mode, variables);
}
#endif
#endif