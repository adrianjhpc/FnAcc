program matmul_2d_f64_openacc
  use bench_utils
  implicit none

  integer(8) :: n, m, k
  integer(8) :: i, j, p
  integer :: reps, r, errors
  real(8), allocatable :: a(:, :), b(:, :), c(:, :)
  real(8) :: acc
  real(8) :: expected
  real(8) :: t0, t1, elapsed
  real(8) :: flops_per_rep
  real(8) :: diff, tol

  call parse_i64_arg(1, 512_8, n)
  call parse_i64_arg(2, 512_8, m)
  call parse_i32_arg(3, 20, reps)

  ! Script-compatible choice:
  !
  !   executable n m reps
  !
  ! Use k = n.
  k = n

  allocate(a(n, k), b(k, m), c(n, m))

  do p = 1, k
    do i = 1, n
      a(i, p) = 0.001 * real(mod(i - 1 + p - 1, 100_8), 8)
    end do
  end do

  do j = 1, m
    do p = 1, k
      b(p, j) = 0.002 * real(mod(p - 1 + 2_8 * (j - 1), 100_8), 8)
    end do
  end do

  c = 0.0

  !$acc enter data copyin(a, b) create(c)

  !$acc parallel loop collapse(2) present(a, b, c) private(acc, p)
  do j = 1, m
    do i = 1, n
      acc = 0.0
      !$acc loop seq
      do p = 1, k
        acc = acc + a(i, p) * b(p, j)
      end do
      c(i, j) = acc
    end do
  end do

  !$acc wait

  t0 = wall_time()

  do r = 1, reps
    !$acc parallel loop collapse(2) present(a, b, c) private(acc, p)
    do j = 1, m
      do i = 1, n
        acc = 0.0
        !$acc loop seq
        do p = 1, k
          acc = acc + a(i, p) * b(p, j)
        end do
        c(i, j) = acc
      end do
    end do
  end do

  !$acc wait
  t1 = wall_time()

  !$acc update self(c)
  !$acc exit data delete(a, b, c)

  errors = 0

  do j = 1, m
    do i = 1, n
      expected = 0.0
      do p = 1, k
        expected = expected + a(i, p) * b(p, j)
      end do

      diff = abs(c(i, j) - expected)
      tol = 1.0e-2 + 1.0e-3 * abs(expected)

      if (diff > tol) then
        if (errors < 10) then
          write(*,*) "mismatch at ", i, j, " got ", c(i, j), &
                     " expected ", expected, " diff ", diff, " tol ", tol
        end if
        errors = errors + 1
      end if
    end do
  end do

  elapsed = t1 - t0

  ! This is FLOPs, not bytes. The existing print_result routine will label
  ! the final rate like the other benchmarks, but for matmul interpret it
  ! as GFLOP/s.
  flops_per_rep = 2.0d0 * real(n, 8) * real(m, 8) * real(k, 8)

  call print_result("openacc_matmul_2d_f64", n, m, reps, elapsed, flops_per_rep, errors)

end program

