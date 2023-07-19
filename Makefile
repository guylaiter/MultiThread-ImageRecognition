build:
	mpicxx -fopenmp -c main.c -o main.o -lm -g
	mpicxx -I/usr/include/x86_64-linux-gnu/mpich -fopenmp -c helper.c -o helper.o -lm -g
	nvcc -I/usr/include/x86_64-linux-gnu/mpich -I./Common -gencode arch=compute_61,code=sm_61 -c cudaHelper.cu -o cudaHelper.o -lm -g
	mpicxx -fopenmp -o final_project_exe main.o helper.o cudaHelper.o -lm -lcudart -L/usr/local/cuda/lib64 -g

clean:
	rm -f *.o ./final_project_exe

run:
	mpiexec -np 2 ./final_project_exe

runOn2:
	mpiexec -np 2 -machinefile mf -map-by node ./final_project_exe

run_debug:
	mpiexec -np 2 xterm -e gdb ./final_project_exe
