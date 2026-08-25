program axpby_fnacc
  use bench_utils
  use fnacc_axpby_kernel
  implicit none

  integer(8) :: n
  integer :: reps
  real, allocatable :: a(:), b(:), c(:)
  real :: alpha, beta, expected
  integer(8) :: i
  integer :: r, errors
  real(8) :: t0, t1, elapsed
  real(8) :: bytes_per_rep

  call parse_i64_arg(1, 1048576_8, n)
  call parse_i32_arg(2, 100, reps)

  alpha = 2.0
  beta = 5.0

  allocate(a(n), b(n), c(n))

  do i = 1, n
    a(i) = real(i)
    b(i) = 100.0 - 0.25 * real(i)
    c(i) = 0.0
  end do

  call axpby_prepare(a, b, c)

  call axpby_compute(alpha, beta, a, b, c)
  call axpby_compute(alpha, beta, a, b, c)

  t0 = wall_time()
  do r = 1, reps
    call axpby_compute(alpha, beta, a, b, c)
  end do
  t1 = wall_time()

  call axpby_fetch(c)

  errors = 0
  do i = 1, n
    expected = alpha * a(i) + beta * b(i)
    if (.not. almost_equal_f32(c(i), expected)) errors = errors + 1
  end do

  call axpby_release(a, b, c)

  elapsed = t1 - t0
  bytes_per_rep = 3.0d0 * real(n, 8) * real(storage_size(a(1))/8, 8)

  call print_result("fnacc_axpby", n, 1_8, reps, elapsed, bytes_per_rep, errors)
end program

