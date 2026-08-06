program main
  use fnacc_assumed_shape_matrix_kernels
  implicit none

  integer, parameter :: n = 64
  integer, parameter :: m = 48
  real, allocatable :: a(:, :), b(:, :), c(:, :)
  real :: expected
  integer :: i, j
  integer :: errors

  allocate(a(n, m), b(n, m), c(n, m))

  do j = 1, m
    do i = 1, n
      a(i, j) = real(i) + 10.0 * real(j)
      b(i, j) = 2.0 * real(i) - real(j)
      c(i, j) = 0.0
    end do
  end do

  call matrix_add_assumed_shape(a, b, c)

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
    print *, "assumed-shape-matrix-add-2d: PASS"
  else
    print *, "assumed-shape-matrix-add-2d: FAIL errors =", errors
    stop 1
  end if
end program

