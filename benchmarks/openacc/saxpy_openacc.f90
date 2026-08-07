program saxpy_openacc
  use bench_utils
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
    x(i) = real(i)
    y(i) = 10.0 + real(i)
    y0(i) = y(i)
  end do

  !$acc enter data copyin(x, y)

  !$acc parallel loop present(x, y)
  do i = 1, n
    y(i) = alpha * x(i) + y(i)
  end do

  !$acc update self(y)

  do i = 1, n
    y(i) = y0(i)
  end do

  !$acc update device(y)

  t0 = wall_time()
  do r = 1, reps
    !$acc parallel loop present(x, y)
    do i = 1, n
      y(i) = alpha * x(i) + y(i)
    end do
  end do
  !$acc wait
  t1 = wall_time()

  !$acc update self(y)
  !$acc exit data delete(x, y)

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

  call print_result("openacc_saxpy", n, 1_8, reps, elapsed, bytes_per_rep, errors)
end program

