program reduction_openacc
  use bench_utils
  implicit none

  integer(8) :: n, i
  integer :: reps, r, errors
  real, allocatable :: x(:), y(:)
  real :: result
  real(8) :: diff, tol
  real :: expected
  real(8) :: t0, t1, elapsed, bytes_per_rep

  call parse_i64_arg(1, 1048576_8, n)
  call parse_i32_arg(2, 100, reps)

  allocate(x(n), y(n))

  do i = 1, n
    x(i) = 0.001 * real(mod(i, 1000_8))
    y(i) = 0.002 * real(mod(3_8 * i, 1000_8))
  end do

  result = 0.0d0

  !$acc enter data copyin(x(1:n), y(1:n), result)

  !$acc parallel loop present(x, y) reduction(+:result)
  do i = 1, n
    result = result + x(i) * y(i)
  end do

  !$acc wait
  t1 = wall_time()

  !$acc update self(result)
  !$acc exit data delete(x(1:n), y(1:n))

  expected = 0.0d0
  do i = 1, n
    expected = expected + real(x(i), 8) * real(y(i), 8)
  end do

  diff = abs(real(result, 8) - expected)
  tol = 1.0d-2 + 5.0d-4 * abs(expected)

  errors = 0
  if (diff > tol) then
    write(*,*) "reduction mismatch got ", result, " expected ", expected, &
               " diff ", diff, " tol ", tol
    errors = 1
  end if

  call print_result("openacc_reduction", n, 1_8, reps, elapsed, bytes_per_rep, errors)
end program

