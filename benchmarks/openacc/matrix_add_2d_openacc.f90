program matrix_add_2d_openacc
  use bench_utils
  implicit none

  integer(8) :: n, m
  integer(8) :: i, j
  integer :: reps, r, errors
  real, allocatable :: a(:, :), b(:, :), c(:, :)
  real :: expected
  real(8) :: t0, t1, elapsed, bytes_per_rep

  call parse_i64_arg(1, 1024_8, n)
  call parse_i64_arg(2, 1024_8, m)
  call parse_i32_arg(3, 100, reps)

  allocate(a(n, m), b(n, m), c(n, m))

  do j = 1, m
    do i = 1, n
      a(i, j) = real(i) + 10.0 * real(j)
      b(i, j) = 2.0 * real(i) - real(j)
      c(i, j) = 0.0
    end do
  end do

  !$acc enter data copyin(a, b) create(c)

  !$acc parallel loop collapse(2) present(a, b, c)
  do j = 1, m
    do i = 1, n
      c(i, j) = a(i, j) + b(i, j)
    end do
  end do

  t0 = wall_time()
  do r = 1, reps
    !$acc parallel loop collapse(2) present(a, b, c)
    do j = 1, m
      do i = 1, n
        c(i, j) = a(i, j) + b(i, j)
      end do
    end do
  end do
  !$acc wait
  t1 = wall_time()

  !$acc update self(c)
  !$acc exit data delete(a, b, c)

  errors = 0
  do j = 1, m
    do i = 1, n
      expected = a(i, j) + b(i, j)
      if (.not. almost_equal_f32(c(i, j), expected)) errors = errors + 1
    end do
  end do

  elapsed = t1 - t0
  bytes_per_rep = 3.0d0 * real(n * m, 8) * 4.0d0

  call print_result("openacc_matrix_add_2d", n, m, reps, elapsed, bytes_per_rep, errors)
end program

