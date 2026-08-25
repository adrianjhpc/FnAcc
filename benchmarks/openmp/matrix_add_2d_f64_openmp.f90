program matrix_add_2d_f64_openmp
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

  do j = 1, m
    do i = 1, n
      a(i, j) = real(i, 8) + 10.0 * real(j, 8)
      b(i, j) = 2.0 * real(i, 8) - real(j, 8)
      c(i, j) = 0.0
    end do
  end do

  !$omp parallel do collapse(2)
  do j = 1, m
    do i = 1, n
      c(i, j) = a(i, j) + b(i, j)
    end do
  end do

  t0 = wall_time()
  !$omp parallel private(r, i, j)
  do r = 1, reps
    !$omp do collapse(2)
    do j = 1, m
      do i = 1, n
        c(i, j) = a(i, j) + b(i, j)
      end do
    end do
  end do
  !$omp end parallel
  t1 = wall_time()

  errors = 0
  do j = 1, m
    do i = 1, n
      expected = a(i, j) + b(i, j)
      if (.not. almost_equal_f64(c(i, j), expected)) errors = errors + 1

    end do
  end do

  elapsed = t1 - t0
  bytes_per_rep = 3.0d0 * real(n * m, 8) * real(storage_size(a(1,1))/8, 8)

  call print_result("openmp_matrix_add_2d_f64", n, m, reps, elapsed, bytes_per_rep, errors)
end program

