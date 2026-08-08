module fnacc_matmul_2d_kernel
contains

  subroutine matmul_2d_prepare(a, b, c)
    real :: a(:, :), b(:, :), c(:, :)

    !$fnacc update device(a)
    !$fnacc update device(b)
    !$fnacc update device(c)
  end subroutine

  subroutine matmul_2d_compute(a, b, c)
    real :: a(:, :), b(:, :), c(:, :)
    integer :: i, j, p
    real :: acc

    !$fnacc parallel tile(16, 16, 32) pack(a:device, b:device, c:device)
    do j = 1, size(c, 2)
      do i = 1, size(c, 1)
        acc = 0.0
        do p = 1, size(a, 2)
          acc = acc + a(i, p) * b(p, j)
        end do
        c(i, j) = acc
      end do
    end do

  end subroutine

  subroutine matmul_2d_fetch(c)
    real :: c(:, :)

    !$fnacc update host(c)
  end subroutine

  subroutine matmul_2d_release(a, b, c)
    real :: a(:, :), b(:, :), c(:, :)

    !$fnacc release(a, b, c)
  end subroutine

end module

