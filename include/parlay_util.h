#include <parlay/parallel.h>

template<typename Range, typename BinaryOp>
auto par_reduce(const Range& A, const BinaryOp&& binop, long block_size = 100) {
	if (block_size == 0) block_size = -1;
	long n = A.size();
	using T = typename BinaryOp::T;
	if (n == 0) return binop.identity;
	if (n <= block_size) {
		T v = binop.identity;
		for (long i = 0; i < n; i++)
			v = binop(v, A[i]);
		return v;
	}

	T L, R;
	parlay::par_do([&] {L = reduce(parlay::make_slice(A).cut(0, n / 2), binop);},
			[&] {R = reduce(parlay::make_slice(A).cut(n / 2, n), binop);});
	return binop(L, R);
}
