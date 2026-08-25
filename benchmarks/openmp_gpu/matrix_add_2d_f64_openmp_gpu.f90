program matrix_add_2d_f64_openmp_gpu
  use bench_utils
  use omp_lib
  implicit none

  integer(8) :: n, m
  integer(8) :: i, j
  integer :: reps, r, errors
  real(8), allocatable :: a(:, :), b(:, :), c(:, :)
  real(8) :: expected
  real(8) :: t0, t1, elapsed, bytes_per_rep

  call parse_i64_arg(1, 1024_8, n)
  call parse_i64_arg(2, 1024_8, m)
  call parse_i32_arg(3, 100, reps)

  allocate(a(n, m), b(n, m), c(n, m))

  ! Initialize data on the CPU
  do j = 1, m
    do i = 1, n
      a(i, j) = real(i, 8) + 10.0 * real(j, 8)
      b(i, j) = 2.0 * real(i, 8) - real(j, 8)
      c(i, j) = 0.0
    end do
  end do

  ! Create data region on the GPU
  ! Map the entire 2D arrays. Specifying bounds (1:n, 1:m) is good practice.
  !$omp target data map(to: a(1:n, 1:m), b(1:n, 1:m)) map(from: c(1:n, 1:m))

  ! Warm-up execution on the GPU
  ! The collapse(2) clause fuses the two loops before distributing them
  !$omp target teams distribute parallel do collapse(2)
  do j = 1, m
    do i = 1, n
      c(i, j) = a(i, j) + b(i, j)
    end do
  end do

  ! Start timing only the GPU execution
  t0 = wall_time()
  
  ! Launch the GPU kernel 'reps' times
  do r = 1, reps
    !$omp target teams distribute parallel do collapse(2)
    do j = 1, m
      do i = 1, n
        c(i, j) = a(i, j) + b(i, j)
      end do
    end do
  end do
  
  t1 = wall_time()

  !$omp end target data
  ! 'c' is copied back to CPU memory here

  ! Verify results on the CPU
  errors = 0
  do j = 1, m
    do i = 1, n
      expected = a(i, j) + b(i, j)
      if (.not. almost_equal_f64(c(i, j), expected)) errors = errors + 1
    end do
  end do

  elapsed = t1 - t0
  bytes_per_rep = 3.0d0 * real(n * m, 8) * real(storage_size(a(1,1))/8, 8)

  call print_result("openmp_target_matrix_add_2d_f64", n, m, reps, elapsed, bytes_per_rep, errors)
end program
