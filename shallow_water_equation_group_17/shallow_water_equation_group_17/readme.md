**Configuration of folder**

          |-- Lax_friedrich_fd.cu
--- CUDA--|-- Lax_friedrich_fv.cu
          |-- Lax_wendroff_fd.cu

          |-- Lax_friedrich_fd.c 
--- MPI --|-- Lax_friedrich_fv.c
          |-- Lax_wendroff_fd.c

                 |-- Lax_friedrich_fd.c (finite different method with Lax_friedrich scheme)
--- Sequential --|-- Lax_friedrich_fv.c (finite volume method with Lax_friedrich scheme)
                 |-- Lax_wendroff_fd.c  (finite different method with Lax_wendroff scheme)