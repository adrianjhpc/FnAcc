program saxpy_openmp_gpu
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

  ! Initialize data on the host
  do i = 1, n
    x(i) = real(i)
    y(i) = 10.0 + real(i)
    y0(i) = y(i)
  end do

  ! Warm-up execution on the GPU
  !$omp target data map(to: x(1:n), y(1:n)) map(from: y(1:n))
  !$omp target teams distribute parallel do
  do i = 1, n
    y(i) = alpha * x(i) + y(i)
  end do
  !$omp end target data

  ! Reset 'y' back to initial state on the CPU
  do i = 1, n
    y(i) = y0(i)
  end do

  ! Map 'x' and 'y' to the device, and retrieve 'y' at the end.
  !$omp target data map(to: x(1:n), y(1:n), alpha) map(from: y(1:n))
  
  t0 = wall_time()
  
  do r = 1, reps
    !$omp target teams distribute parallel do
    do i = 1, n
      y(i) = alpha * x(i) + y(i)
    end do
  end do
  
  t1 = wall_time()
  
  !$omp end target data

  errors = 0
  do i = 1, n
    expected = y0(i)
    do r = 1, reps
      expected = alpha * x(i) + expected
    end do

    if (.not. almost_equal_f32(y(i), expected)) errors = errors + 1
  end do

  elapsed = t1 - t0
  bytes_per_rep = 3.0d0 * real(n, 8) * 4.0d0

  call print_result("openmp_target_saxpy", n, 1_8, reps, elapsed, bytes_per_rep, errors)
end program
