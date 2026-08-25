program daxpy_fnacc
  use bench_utils
  use fnacc_daxpy_kernel
  implicit none

  integer(8) :: n
  integer :: reps
  real(8), allocatable :: x(:), y(:), y0(:)
  real(8) :: alpha
  real(8) :: expected
  integer(8) :: i
  integer :: r, errors
  real(8) :: t0, t1, elapsed
  real(8) :: bytes_per_rep

  call parse_i64_arg(1, 1048576_8, n)
  call parse_i32_arg(2, 100, reps)

  alpha = 3.0

  allocate(x(n), y(n), y0(n))

  do i = 1, n
    x(i) = 0.001 * real(mod(i, 1000_8), 8)
    y(i) = 10.0 + 0.002 * real(mod(i, 1000_8), 8)
    y0(i) = y(i)
  end do

  call daxpy_prepare(x, y)

  call daxpy_compute(alpha, x, y)
  call daxpy_fetch(y)

  y(:) = y0(:)
  call daxpy_prepare(x, y)

  t0 = wall_time()
  do r = 1, reps
    call daxpy_compute(alpha, x, y)
  end do
  t1 = wall_time()

  call daxpy_fetch(y)

  errors = 0
  do i = 1, n
    expected = y0(i) + real(reps, 8) * alpha * x(i)
    if (.not. almost_equal_f64(y(i), expected)) errors = errors + 1
  end do

  call daxpy_release(x, y)

  elapsed = t1 - t0
  bytes_per_rep = 3.0d0 * real(n, 8) * real(storage_size(y(1))/8, 8)

  call print_result("fnacc_daxpy", n, 1_8, reps, elapsed, bytes_per_rep, errors)
end program

