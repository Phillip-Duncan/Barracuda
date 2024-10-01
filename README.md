# Barracuda

A simple header-only pseudo-stack-based equation solver capable of running on the GPU in 'effective' stacksize-linear time. Evaluates inversely stored reverse polish notation (RPN) instructions from input stacks.

## Inputs

### Required

stack: an input integer instruction stack corresponding to whether a value or an operation is executed. Operations can also consist of loading (storing) arbitrary variables to the output stack. The stack is evaluated ("pop"-ed) last-to-first.
stacksize: size of the instruction stack.

opstack: stack containing integer values of operation instructions corresponding to the enumerator types (OPCODES).
opstacksize: size of the opstack. The opstack is evaluated ("pop"-ed) last-to-first.

valuestack: stack containing floating point values to be loaded to the output stack.
valuestacksize: size of the valuestack. The valuestack is evaluated ("pop"-ed) last-to-first.

outputstack: stack storing the "outputs" of applied operations or stores for future use. Operations from the opstack apply directly to the outputstack.  I.e. to perform multiplication of two variables, first load these into the output stack and then apply the OPCODE (0x3CE) in the opstack.

### Optional

user_space: if implemented via memory allocation/copy, allows users to use memory in the form of arrays. Some form of user_space should always be allocated for environment variables (if used) regardless, the "user" part of the user_space is the extension of the user_space past the environment variables to the maximum size of the number and sizes of allocated arrays by the user in their program code.

## Written Examples

A = 2, B = 3, C = 4

expr = A \* B + C

expr(RPN): AB\*C+

```cpp
stack[5] = {0,1,0,1,1}
opstack[5] = {0x3CC,0,0x3CE,0,0};
valuestack[5] = {0,4,0,3,2};
```

The above code evaluates to:  
push(2)  --> A  
push(3)  --> B  
v = MUL(2,3) --> A\*B  
push(v) --> A\*B  
push(4) --> C  
v = ADD(6,4) --> A\*B + C  
push(v) --> A\*B + C  

when MUL, ADD or any operation are used, pop(outputstack) is called an appropriate number of times relative to the number of inputs the function requires.

## Provided Examples

All previous examples have now been deprecated due to several significant changes and the development of an open-source compiler for Barracuda. Due to this change the primary example of Barracuda is now vm_generic, which compiles a generic form standalone of the Barracuda VM (with two example environment variables set for testing). This generic VM is provided to allow for testing of arbitrary compiled Barracuda code from the rust barracuda compiler. A testing framework for this has been made in Python to implement tests that run both the compiler and the VM for end-to-end testing and validation.
