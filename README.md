# HUST-ParallelProgramming-HeatEquation

## Project Description

Parallel programming is a key approach for speeding up numerical algorithms. This pa-
per focuses on parallelizing a program designed to solve the two-dimensional shallow water
equation, which serves as an example of a conservation law. The equation is tackled using finite
difference and finite volume techniques, applying schemes such as the Lax-Friedrichs and
Lax-Wendroff methods (we exploit 3 algorithms). The parallel implementation employs both
MPI and CUDA. Performance evaluation is primarily conducted on shared memory systems,
and due to the availability of a sufficient number of processors, results are also presented for
MPI and CUDA

## MPI

![image](https://github.com/user-attachments/assets/303e3934-cc30-4557-9f24-7e05514bb2ae)


## CUDA

![image](https://github.com/user-attachments/assets/45101d10-2acf-4595-a0cf-68cce0e26e23)

## 🧪 Results and Simulations
<img width="609" height="785" alt="image" src="https://github.com/user-attachments/assets/a191afa8-fb89-4b99-affb-15857778aba3" />
<img width="604" height="799" alt="image" src="https://github.com/user-attachments/assets/5bebf9ce-74ea-4ffc-9da2-c89a280a4a49" />
<img width="595" height="547" alt="image" src="https://github.com/user-attachments/assets/f5b9422f-23cf-4ec0-b52e-719dd02dd54b" />
<img width="555" height="794" alt="image" src="https://github.com/user-attachments/assets/2f87972a-1944-4259-a76a-6c13d2811bdd" />
