program main
  implicit none

  interface
    subroutine compute_axpby(n, alpha, beta, a, b, c)
      integer :: n
      real :: alpha, beta
      real :: a(n), b(n), c(n)
    end subroutine
  end interface

  integer, parameter :: n = 1024
  real, parameter :: alpha = 2.0
  real, parameter :: beta = 5.0
  real :: a(n), b(n), c(n)
  real :: expected
  integer :: i
  integer :: errors

  do i = 1, n
    a(i) = real(i)
    b(i) = 100.0 - real(i) * 0.25
    c(i) = 0.0
  end do

  call compute_axpby(n, alpha, beta, a, b, c)

  errors = 0
  do i = 1, n
    expected = alpha * a(i) + beta * b(i)
    if (abs(c(i) - expected) > 1.0e-4) then
      if (errors < 10) then
        print *, "mismatch at", i, "got", c(i), "expected", expected
      end if
      errors = errors + 1
    end if
  end do

  if (errors == 0) then
    print *, "axpby: PASS"
  else
    print *, "axpby: FAIL errors =", errors
    stop 1
  end if
end program
