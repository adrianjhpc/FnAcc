subroutine compute_saxpy(n, alpha, x, y)
  integer :: n
  real :: alpha
  real :: x(n), y(n)
  integer :: i

  !$fnacc parallel tile(128) pack(x:device, y:device)
  do i = 1, n
    y(i) = alpha * x(i) + y(i)
  end do

  !$fnacc update host(y)
  !$fnacc release all
end subroutine
