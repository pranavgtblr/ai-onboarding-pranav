from phase_0_baseline import main


def test_main(capsys: object) -> None:
    main()
    # verify main runs without crashing
    assert True
