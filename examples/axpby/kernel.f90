subroutine compute_axpby(n, alpha, beta, a, b, c)
  integer :: n
  real :: alpha, beta
  real :: a(n), b(n), c(n)
  integer :: i

  !$fnacc parallel tile(128) pack(a:device, b:device, c:device)
  do i = 1, n
    c(i) = alpha * a(i) + beta * b(i)
  end do

  !$fnacc update host(c)
  !$fnacc release all
end subroutine
