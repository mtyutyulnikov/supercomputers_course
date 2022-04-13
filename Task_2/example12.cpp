#include <omp.h>
#include <iostream>
using namespace std;
int main(int argc, char **argv) {
	const int size = (argc > 1 ? atoi(argv[1]) : 1000);
	int *a = new int[size];
	#pragma omp parallel for
	for(int i = 0; i < size; i++) {
		a[i] = i;
	}
	int sum = 0;
	#pragma omp parallel for reduction(+:sum)
	for(int i = 0; i < size; i++) {
		sum += a[i];
	}
	cout << "Sum = " << sum << endl;
	delete[] a;
	return 0;
}
