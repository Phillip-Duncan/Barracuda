## 1.1.0 (2025-03-26)

### Feat

- **stack.cuh,-barracuda.cuh,-vm_generic.cu,-vm_generic.cuh**: Add in two new instructions (LDCUX, LDCUPTR) for loading in constant memory values and the constant memory pointer. This required adding in a new input (userspace_sizes) that gives the VM the sizes of the mutable userspace and constant userspace respectively so that it can appropriately jump to the proper memory locations and do bounds checking. The vm_generic program has been updated to reflect these changes
