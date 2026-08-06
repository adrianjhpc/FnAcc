program main
  use fnacc_assumed_shape_kernels
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

  call vector_add_assumed_shape(a, b, c)

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
    print *, "assumed-shape-vector-add: PASS"
  else
    print *, "assumed-shape-vector-add: FAIL errors =", errors
    stop 1
  end if
end program

