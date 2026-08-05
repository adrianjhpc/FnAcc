subroutine compute_matrix_add(n, m, a, b, c)
  integer :: n, m
  real :: a(n, m), b(n, m), c(n, m)
  integer :: i, j

  !$fnacc parallel tile(16, 16) pack(a:device, b:device, c:device)
  do j = 1, m
    do i = 1, n
      c(i, j) = a(i, j) + b(i, j)
    end do
  end do

  !$fnacc update host(c)
  !$fnacc release all
end subroutine
