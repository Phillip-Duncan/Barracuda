# MathStack

A simple header-only stack-based equation solver capable of running on the GPU in stacksize-linear time. Evaluates inversely stored reverse polish notation (RPN) instructions from input stacks.

## Required Inputs

stack: an input integer instruction stack corresponding to whether a value or an operation is executed. Operations can also consist of loading (storing) arbitrary variables to the output stack. The stack is evaluated ("pop"-ed) last-to-first.
stacksize: size of the instruction stack.

opstack: stack containing integer values of operation instructions corresponding to the enumerator types (OPCODES).
opstacksize: size of the opstack. The opstack is evaluated ("pop"-ed) last-to-first.

valuestack: stack containing floating point values to be loaded to the output stack.
valuestacksize: size of the valuestack. The valuestack is evaluated ("pop"-ed) last-to-first.

outputstack: stack storing the "outputs" of applied operations or stores for future use. Operations from the opstack apply directly to the outputstack.  I.e. to perform multiplication of two variables, first load these into the output stack and then apply the OPCODE (0x3CE) in the opstack.

## Written Examples

A = 2, B = 3, C = 4

expr = A \* B + C

expr(RPN): AB\*C+

```cpp
stack[5] = {0,1,0,1,1}
opstack[2] = {0x3CC, 0x3CE};
valuestack[3] = {4,3,2};
```

The above code evaluates to:  
push(2)  --> A  
push(3)  --> B  
v = mult(2,3) --> A\*B  
push(v) --> A\*B  
push(4) --> C  
v = add(6,4) --> A\*B + C  
push(v) --> A\*B + C  

when mult, add or any operation are used, pop(outputstack) is called an appropriate number of times relative to the number of inputs the function requires.

## Provided Examples

example1: Simple example with 256 threads, 256 blocks repeating an iteration 50,000 times (1000 loops within each kernel instance and 50 instances of the kernel itself) of the expression { 10/(sin(6+5)\*1.5\*1.5) }, or in RPN notation { 10 6 5 + sin 1.5 \* 1.5 \*  / }.

example2: Extension of example 1 but using the function pointer of atan2 as the final operation rather than the divide (and only looping over a single iteration). RPN notation of this expr is { 10 6 5 + sin 1.5 \* 1.5 \*  atan2 }.

example2: Complex example solving an integral of the user-defined function sinh(x)cos(x) between [0,pi], with a maxstep of 1000, and desired accuracy of 0.0001.
