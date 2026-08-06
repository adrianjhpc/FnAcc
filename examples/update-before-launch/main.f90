program main
  use fnacc_update_before_launch_kernels
  implicit none

  integer, parameter :: n = 1024
  real, allocatable :: a(:), b(:), c(:)
  real :: expected
  integer :: i
  integer :: errors

  allocate(a(n), b(n), c(n))

  do i = 1, n
    a(i) = real(i)
    b(i) = 2.0 * real(i)
    c(i) = 0.0
  end do

  call update_then_launch(a, b, c)

  errors = 0
  do i = 1, n
    expected = a(i) + b(i)
    if (abs(c(i) - expected) > 1.0e-5) then
      if (errors < 10) then
        print *, "mismatch at", i, "got", c(i), "expected", expected
      end if
      errors = errors + 1
    end if
  end do

  if (errors == 0) then
    print *, "update-before-launch: PASS"
  else
    print *, "update-before-launch: FAIL errors =", errors
    stop 1
  end if
end program

