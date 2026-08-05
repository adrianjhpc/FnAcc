program main
  implicit none

  interface
    subroutine compute_saxpy(n, alpha, x, y)
      integer :: n
      real :: alpha
      real :: x(n), y(n)
    end subroutine
  end interface

  integer, parameter :: n = 1024
  real, parameter :: alpha = 3.0
  real :: x(n), y(n), y0(n)
  real :: expected
  integer :: i
  integer :: errors

  do i = 1, n
    x(i) = real(i)
    y(i) = 10.0 + real(i)
    y0(i) = y(i)
  end do

  call compute_saxpy(n, alpha, x, y)

  errors = 0
  do i = 1, n
    expected = alpha * x(i) + y0(i)
    if (abs(y(i) - expected) > 1.0e-5) then
      if (errors < 10) then
        print *, "mismatch at", i, "got", y(i), "expected", expected
      end if
      errors = errors + 1
    end if
  end do

  if (errors == 0) then
    print *, "saxpy: PASS"
  else
    print *, "saxpy: FAIL errors =", errors
    stop 1
  end if
end program
