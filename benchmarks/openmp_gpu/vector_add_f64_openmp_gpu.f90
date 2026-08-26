program vector_add_f64_openmp_gpu
  use bench_utils
  use omp_lib
  implicit none

  integer(8) :: n, i
  integer :: reps, r, errors
  real(8), allocatable :: a(:), b(:), c(:)
  real(8) :: expected
  real(8) :: t0, t1, elapsed, bytes_per_rep

  call parse_i64_arg(1, 1048576_8, n)
  call parse_i32_arg(2, 100, reps)

  allocate(a(n), b(n), c(n))

  ! Initialize data on the CPU
  do i = 1, n
    a(i) = real(i, 8)
    b(i) = 2.0 * real(i, 8)
    c(i) = 0.0
  end do

  ! Map 'a' and 'b' to the GPU, and 'c' back to the CPU at the end
  !$omp target data map(to: a(1:n), b(1:n)) map(from: c(1:n))

  ! Warm-up execution on the GPU
  !$omp target teams distribute parallel do
  do i = 1, n
    c(i) = a(i) + b(i)
  end do

  ! Start timing only the GPU execution
  t0 = wall_time()
  
  ! Launch the GPU kernel 'reps' times
  do r = 1, reps
    !$omp target teams distribute parallel do
    do i = 1, n
      c(i) = a(i) + b(i)
    end do
  end do
  
  t1 = wall_time()

  !$omp end target data
  ! 'c' is copied back to CPU memory here

  ! Verify results on the CPU
  errors = 0
  do i = 1, n
    expected = a(i) + b(i)
    if (.not. almost_equal_f64(c(i), expected)) errors = errors + 1
  end do

  elapsed = t1 - t0
  bytes_per_rep = 3.0d0 * real(n, 8) * real(storage_size(c(1))/8, 8)

  call print_result("openmp_target_vector_add_f64", n, 1_8, reps, elapsed, bytes_per_rep, errors)
end program
