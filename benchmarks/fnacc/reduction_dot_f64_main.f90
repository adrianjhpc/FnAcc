program reduction_dot_f64_fnacc
  use bench_utils
  use fnacc_reduction_dot_f64_kernel
  implicit none

  integer(8) :: n
  integer :: reps
  real(8), allocatable :: a(:), b(:)
  real(8) :: result
  real(8) :: expected
  real(8) :: diff, tol
  integer(8) :: i
  integer :: r, errors
  real(8) :: t0, t1, elapsed
  real(8) :: bytes_per_rep

  call parse_i64_arg(1, 1048576_8, n)
  call parse_i32_arg(2, 100, reps)

  allocate(a(n), b(n))

  do i = 1, n
    a(i) = 0.001_8 * real(mod(i, 1000_8), 8)
    b(i) = 0.002_8 * real(mod(3_8 * i, 1000_8), 8)
  end do

  call reduction_dot_f64_prepare(a, b)

  call reduction_dot_f64_compute(a, b, result)
  call reduction_dot_f64_compute(a, b, result)

  t0 = wall_time()
  do r = 1, reps
    call reduction_dot_f64_compute(a, b, result)
  end do
  t1 = wall_time()

  expected = 0.0_8
  do i = 1, n
    expected = expected + a(i) * b(i)
  end do

  diff = abs(result - expected)
  tol = 1.0d-8 + 1.0d-10 * abs(expected)

  errors = 0
  if (diff > tol) then
    write(*,*) "f64 dot mismatch got ", result, " expected ", expected, &
               " diff ", diff, " tol ", tol
    errors = 1
  end if

  call reduction_dot_f64_release(a, b)

  elapsed = t1 - t0
  bytes_per_rep = 2.0d0 * real(n, 8) * 8.0d0

  call print_result("fnacc_reduction_dot_f64", n, 1_8, reps, elapsed, bytes_per_rep, errors)
end program

