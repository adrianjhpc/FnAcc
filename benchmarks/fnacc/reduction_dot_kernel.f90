module fnacc_reduction_dot_kernel
contains

  subroutine reduction_dot_prepare(a, b)
    real :: a(:), b(:)

    !$fnacc enter data copyin(a, b)
  end subroutine

  subroutine reduction_dot_compute(a, b, result)
    real :: a(:), b(:)
    real :: result
    integer :: i

    result = 0.0

    !$fnacc parallel tile(256) reduction(+:result)
    do i = 1, size(a)
      result = result + a(i) * b(i)
    end do
  end subroutine

  subroutine reduction_dot_release(a, b)
    real :: a(:), b(:)

    !$fnacc exit data delete(a, b)
  end subroutine

end module

