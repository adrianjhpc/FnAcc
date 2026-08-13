module fnacc_reduction_dot_f64_kernel
contains

  subroutine reduction_dot_f64_prepare(a, b)
    real(8) :: a(:), b(:)

    !$fnacc enter data copyin(a, b)
  end subroutine

  subroutine reduction_dot_f64_compute(a, b, result)
    real(8) :: a(:), b(:)
    real(8) :: result
    integer :: i

    result = 0.0_8

    !$fnacc parallel tile(256) reduction(+:result)
    do i = 1, size(a)
      result = result + a(i) * b(i)
    end do
  end subroutine

  subroutine reduction_dot_f64_release(a, b)
    real(8) :: a(:), b(:)

    !$fnacc exit data delete(a, b)
  end subroutine

end module

