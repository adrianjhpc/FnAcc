program main
  implicit none

  interface
    subroutine compute_matrix_add(n, m, a, b, c)
      integer :: n, m
      real :: a(n, m), b(n, m), c(n, m)
    end subroutine
  end interface

  integer, parameter :: n = 64
  integer, parameter :: m = 48
  real :: a(n, m), b(n, m), c(n, m)
  real :: expected
  integer :: i, j
  integer :: errors

  do j = 1, m
    do i = 1, n
      a(i, j) = real(i) + 10.0 * real(j)
      b(i, j) = 2.0 * real(i) - real(j)
      c(i, j) = 0.0
    end do
  end do

  call compute_matrix_add(n, m, a, b, c)

  errors = 0
  do j = 1, m
    do i = 1, n
      expected = a(i, j) + b(i, j)
      if (abs(c(i, j) - expected) > 1.0e-5) then
        if (errors < 10) then
          print *, "mismatch at", i, j, "got", c(i, j), "expected", expected
        end if
        errors = errors + 1
      end if
    end do
  end do

  if (errors == 0) then
    print *, "matrix-add-2d: PASS"
  else
    print *, "matrix-add-2d: FAIL errors =", errors
    stop 1
  end if
end program
