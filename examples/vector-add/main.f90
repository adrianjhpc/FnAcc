program main
  implicit none

  interface
    subroutine compute_add(n, a, b, c)
      integer :: n
      real :: a(n), b(n), c(n)
    end subroutine
  end interface

  integer, parameter :: n = 1024
  real :: a(n), b(n), c(n)
  real :: expected
  integer :: i
  integer :: errors

  do i = 1, n
    a(i) = real(i)
    b(i) = 2.0 * real(i)
    c(i) = 0.0
  end do

  call compute_add(n, a, b, c)

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
    print *, "vector-add: PASS"
  else
    print *, "vector-add: FAIL errors =", errors
    stop 1
  end if
end program
