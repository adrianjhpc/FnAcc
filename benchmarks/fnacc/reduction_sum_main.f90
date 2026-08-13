program reduction_sum_fnacc
  use bench_utils
  use fnacc_reduction_sum_kernel
  implicit none

  integer(8) :: n
  integer :: reps
  real, allocatable :: a(:)
  real :: result
  real(8) :: expected
  real(8) :: diff, tol
  integer(8) :: i
  integer :: r, errors
  real(8) :: t0, t1, elapsed
  real(8) :: bytes_per_rep

  call parse_i64_arg(1, 1048576_8, n)
  call parse_i32_arg(2, 100, reps)

  allocate(a(n))

  do i = 1, n
    a(i) = 0.001 * real(mod(i, 1000_8))
  end do

  call reduction_sum_prepare(a)

  call reduction_sum_compute(a, result)
  call reduction_sum_compute(a, result)

  t0 = wall_time()
  do r = 1, reps
    call reduction_sum_compute(a, result)
  end do
  t1 = wall_time()

  expected = 0.0d0
  do i = 1, n
    expected = expected + real(a(i), 8)
  end do

  diff = abs(real(result, 8) - expected)
  tol = 1.0d-2 + 1.0d-4 * abs(expected)

  errors = 0
  if (diff > tol) then
    write(*,*) "sum mismatch got ", result, " expected ", expected, &
               " diff ", diff, " tol ", tol
    errors = 1
  end if

  call reduction_sum_release(a)

  elapsed = t1 - t0
  bytes_per_rep = real(n, 8) * 4.0d0

  call print_result("fnacc_reduction_sum", n, 1_8, reps, elapsed, bytes_per_rep, errors)
end program

