program saxpy_openmp
  use bench_utils
  use omp_lib
  implicit none

  integer(8) :: n, i
  integer :: reps, r, errors
  real, allocatable :: x(:), y(:), y0(:)
  real :: alpha
  real :: expected
  real(8) :: t0, t1, elapsed, bytes_per_rep

  call parse_i64_arg(1, 1048576_8, n)
  call parse_i32_arg(2, 100, reps)

  alpha = 3.0

  allocate(x(n), y(n), y0(n))

  do i = 1, n
    x(i) = 0.001 * real(mod(i, 1000_8))
    y(i) = 10.0 + 0.002 * real(mod(i, 1000_8))
    y0(i) = y(i)
  end do

  !$omp parallel do
  do i = 1, n
    y(i) = alpha * x(i) + y(i)
  end do

  y(:) = y0(:)

  t0 = wall_time()

  !$omp parallel private(r, i)
  do r = 1, reps
    !$omp do
    do i = 1, n
      y(i) = alpha * x(i) + y(i)
    end do
    !$omp end do
  end do
  !$omp end parallel

  t1 = wall_time()

  errors = 0
  do i = 1, n
    expected = y0(i) + real(reps) * alpha * x(i)
    if (.not. almost_equal_f32(y(i), expected)) errors = errors + 1
  end do

  elapsed = t1 - t0
  bytes_per_rep = 3.0d0 * real(n, 8) * 4.0d0

  call print_result("openmp_saxpy", n, 1_8, reps, elapsed, bytes_per_rep, errors)
end program

