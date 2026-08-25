program daxpy_openmp_gpu
  use bench_utils
  use omp_lib
  implicit none

  integer(8) :: n, i
  integer :: reps, r, errors
  real(8), allocatable :: x(:), y(:), y0(:)
  real(8) :: alpha
  real(8) :: expected
  real(8) :: t0, t1, elapsed, bytes_per_rep

  call parse_i64_arg(1, 1048576_8, n)
  call parse_i32_arg(2, 100, reps)

  alpha = 3.0

  allocate(x(n), y(n), y0(n))

  ! Initialize data on the host
  do i = 1, n
    x(i) = 0.001 * real(mod(i, 1000_8), 8)
    y(i) = 10.0 + 0.002 * real(mod(i, 1000_8), 8)
    y0(i) = y(i)
  end do

  ! Warm-up execution on the GPU.
  ! y is read and written, so use tofrom.
  !$omp target data map(to: x(1:n), alpha) map(tofrom: y(1:n))
  !$omp target teams distribute parallel do
  do i = 1, n
    y(i) = alpha * x(i) + y(i)
  end do
  !$omp end target teams distribute parallel do
  !$omp end target data

  ! Reset y back to initial state on the CPU
  do i = 1, n
    y(i) = y0(i)
  end do

  ! Timed execution.
  ! y must be copied to the device initially because the kernel reads y(i).
  !$omp target data map(to: x(1:n)) map(tofrom: y(1:n))

  t0 = wall_time()

  do r = 1, reps
    !$omp target teams distribute parallel do firstprivate(alpha, n)
    do i = 1, n
      y(i) = alpha * x(i) + y(i)
    end do
    !$omp end target teams distribute parallel do
  end do

  t1 = wall_time()

  !$omp end target data

  errors = 0
  do i = 1, n
    expected = y0(i)
    do r = 1, reps
      expected = alpha * x(i) + expected
    end do

    if (.not. almost_equal_f64(y(i), expected)) errors = errors + 1
  end do

  elapsed = t1 - t0
  bytes_per_rep = 3.0d0 * real(n, 8) * real(storage_size(y(1))/8, 8)

  call print_result("openmp_target_daxpy", n, 1_8, reps, elapsed, bytes_per_rep, errors)
end program

