module fnacc_matmul_2d_f64_kernel
contains

  subroutine matmul_2d_f64_prepare(a, b, c)
    real(8) :: a(:, :), b(:, :), c(:, :)

    !$fnacc enter data copyin(a, b) create(c)
  end subroutine

  subroutine matmul_2d_f64_compute(a, b, c)
    real(8) :: a(:, :), b(:, :), c(:, :)
    integer :: i, j, p
    real(8) :: acc

    ! f64 matmul currently uses correctness-first scalar-K fallback.
    !$fnacc parallel tile(32, 16, 8)
    do j = 1, size(c, 2)
      do i = 1, size(c, 1)
        acc = 0.0_8
        do p = 1, size(a, 2)
          acc = acc + a(i, p) * b(p, j)
        end do
        c(i, j) = acc
      end do
    end do
  end subroutine

  subroutine matmul_2d_f64_fetch(c)
    real(8) :: c(:, :)

    !$fnacc exit data copyout(c)
  end subroutine

  subroutine matmul_2d_f64_release(a, b, c)
    real(8) :: a(:, :), b(:, :), c(:, :)

    !$fnacc exit data delete(a, b, c)
  end subroutine

end module

