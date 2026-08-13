module fnacc_reduction_sum_kernel
contains

  subroutine reduction_sum_prepare(a)
    real :: a(:)

    !$fnacc enter data copyin(a)
  end subroutine

  subroutine reduction_sum_compute(a, result)
    real :: a(:)
    real :: result
    integer :: i

    result = 0.0

    !$fnacc parallel tile(256) reduction(+:result)
    do i = 1, size(a)
      result = result + a(i)
    end do
  end subroutine

  subroutine reduction_sum_release(a)
    real :: a(:)

    !$fnacc exit data delete(a)
  end subroutine

end module

