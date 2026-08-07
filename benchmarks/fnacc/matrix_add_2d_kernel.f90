module fnacc_matrix_add_2d_kernel
contains

  subroutine matrix_add_2d_prepare(a, b, c)
    real :: a(:, :), b(:, :), c(:, :)

    !$fnacc update device(a)
    !$fnacc update device(b)
    !$fnacc update device(c)
  end subroutine

  subroutine matrix_add_2d_compute(a, b, c)
    real :: a(:, :), b(:, :), c(:, :)
    integer :: i, j

    !$fnacc parallel tile(16, 16) pack(a:device, b:device, c:device)
    do j = 1, size(c, 2)
      do i = 1, size(c, 1)
        c(i, j) = a(i, j) + b(i, j)
      end do
    end do
  end subroutine

  subroutine matrix_add_2d_fetch(c)
    real :: c(:, :)

    !$fnacc update host(c)
  end subroutine

  subroutine matrix_add_2d_release(a, b, c)
    real :: a(:, :), b(:, :), c(:, :)

    !$fnacc release(a, b, c)
  end subroutine

end module

