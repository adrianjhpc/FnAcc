program matrix_add_2d_f64_fnacc
  use bench_utils
  use fnacc_matrix_add_2d_f64_kernel
  implicit none

  integer(8) :: n, m
  integer :: reps
  real(8), allocatable :: a(:, :), b(:, :), c(:, :)
  real(8) :: expected
  integer(8) :: i, j
  integer :: r, errors
  real(8) :: t0, t1, elapsed
  real(8) :: bytes_per_rep

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

  call matrix_add_2d_f64_prepare(a, b, c)

  call matrix_add_2d_f64_compute(a, b, c)
  call matrix_add_2d_f64_compute(a, b, c)

  t0 = wall_time()
  do r = 1, reps
    call matrix_add_2d_f64_compute(a, b, c)
  end do
  t1 = wall_time()

  call matrix_add_2d_f64_fetch(c)

  errors = 0
  do j = 1, m
    do i = 1, n
      expected = a(i, j) + b(i, j)
      if (.not. almost_equal_f64(c(i, j), expected)) errors = errors + 1

    end do
  end do

  call matrix_add_2d_f64_release(a, b, c)

  elapsed = t1 - t0
  bytes_per_rep = 3.0d0 * real(n * m, 8) * real(storage_size(a(1,1))/8, 8)

  call print_result("fnacc_matrix_add_2d_f64", n, m, reps, elapsed, bytes_per_rep, errors)
end program

