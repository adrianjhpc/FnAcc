program matmul_2d_f64_fnacc
  use bench_utils
  use fnacc_matmul_2d_f64_kernel
  implicit none

  integer(8) :: n, m, k
  integer :: reps
  real(8), allocatable :: a(:, :), b(:, :), c(:, :)
  real(8) :: expected
  real(8) :: diff, tol
  integer(8) :: i, j, p
  integer :: r, errors
  real(8) :: t0, t1, elapsed
  real(8) :: flops_per_rep

  call parse_i64_arg(1, 512_8, n)
  call parse_i64_arg(2, 512_8, m)
  call parse_i32_arg(3, 20, reps)

  k = n

  allocate(a(n, k), b(k, m), c(n, m))

  do p = 1, k
    do i = 1, n
      a(i, p) = 0.001_8 * real(mod(i - 1 + p - 1, 100_8), 8)
    end do
  end do

  do j = 1, m
    do p = 1, k
      b(p, j) = 0.002_8 * real(mod(p - 1 + 2_8 * (j - 1), 100_8), 8)
    end do
  end do

  c = 0.0_8

  call matmul_2d_f64_prepare(a, b, c)

  call matmul_2d_f64_compute(a, b, c)
  call matmul_2d_f64_compute(a, b, c)

  t0 = wall_time()
  do r = 1, reps
    call matmul_2d_f64_compute(a, b, c)
  end do
  t1 = wall_time()

  call matmul_2d_f64_fetch(c)

  errors = 0

  do j = 1, m
    do i = 1, n
      expected = 0.0_8
      do p = 1, k
        expected = expected + a(i, p) * b(p, j)
      end do

      diff = abs(c(i, j) - expected)
      tol = 1.0d-8 + 1.0d-10 * abs(expected)

      if (diff > tol) then
        if (errors < 10) then
          write(*,*) "mismatch at ", i, j, " got ", c(i, j), &
                     " expected ", expected, " diff ", diff, " tol ", tol
        end if
        errors = errors + 1
      end if
    end do
  end do

  call matmul_2d_f64_release(a, b, c)

  elapsed = t1 - t0
  flops_per_rep = 2.0d0 * real(n, 8) * real(m, 8) * real(k, 8)

  call print_result("fnacc_matmul_2d_f64", n, m, reps, elapsed, flops_per_rep, errors)
end program

