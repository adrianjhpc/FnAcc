program axpby_openmp_gpu
  use bench_utils
  use omp_lib
  implicit none

  integer(8) :: n, i
  integer :: reps, r, errors
  real, allocatable :: a(:), b(:), c(:)
  real :: alpha, beta
  real :: expected
  real(8) :: t0, t1, elapsed, bytes_per_rep

  call parse_i64_arg(1, 1048576_8, n)
  call parse_i32_arg(2, 100, reps)

  alpha = 2.0
  beta = 5.0

  allocate(a(n), b(n), c(n))

  ! Initialize data on the host
  do i = 1, n
    a(i) = real(i)
    b(i) = 100.0 - 0.25 * real(i)
    c(i) = 0.0
  end do

  ! Create a data region on the GPU
  ! Map 'a' and 'b' to the device, and retrieve 'c' from the device at the end
  !$omp target data map(to: a(1:n), b(1:n)) map(from: c(1:n))

  ! Warm-up execution on the GPU
  !$omp target teams distribute parallel do
  do i = 1, n
    c(i) = alpha * a(i) + beta * b(i)
  end do

  t0 = wall_time()
  
  ! Main benchmark loop
  do r = 1, reps
    !$omp target teams distribute parallel do
    do i = 1, n
      c(i) = alpha * a(i) + beta * b(i)
    end do
  end do
  
  t1 = wall_time()

  !$omp end target data
  ! At this point, the GPU copies the results of 'c' back to the CPU memory

  errors = 0
  do i = 1, n
    expected = alpha * a(i) + beta * b(i)
    if (.not. almost_equal_f32(c(i), expected)) errors = errors + 1
  end do

  elapsed = t1 - t0
  bytes_per_rep = 3.0d0 * real(n, 8) * 4.0d0

  call print_result("openmp_target_axpby", n, 1_8, reps, elapsed, bytes_per_rep, errors)
end program
