# Changelog

## [v2.2.2] - 2026-06-15

### Fixed
- Fix issue with long joined well names (#82).

### Documentation
- Add changelog file and release-note structure (#81).

## [v2.2.1] - 2026-06-10

### Fixed
- Ensure `well_id` is read as string in file loading to support numeric well identifiers (#72).
- Avoid metrics/sorting issues when wells are identical (#51).

### Added
- Add writing of summed forecasts to file in ADCA workflows (#67).
- Add support for summing forecasts (#66).

### Changed
- Bump release version after well-name fix (#80).
- Update Python support in CI/runtime matrix (3.9 -> 3.9.2, 3.11 -> 3.14) (#53, #54).
- Group and update dependency maintenance process (#68, #69).
- Downgrade `ipython` dependency to resolve compatibility concerns (#71).

### Documentation
- Improve `system_card.md` with human-oversight details (#32).
- Fix typos and spelling issues (#65).

## [v2.1.2] - 2026-05-04

### Fixed
- Skip problematic sorting path for identical wells, then apply a simplified fix (#51).

### Changed
- Bump release to 2.1.2.
- Move project requirements into `pyproject.toml` (#38).
- Split dependencies into dev and doc groups (#39).
- Update `numpy` to >=2.0.2 (#43).
- Update `pillow` to >=11.3.0 (#44).
- Update `pandas` to >=2.3.3 (#46).
- Update `setuptools` to >=82.0.1 (#47).
- Update `ruff` to >=0.15.10 (#49).

### Documentation
- Remove unused sections and fix typos in reporting/docs content (#48).

## [v2.1.1] - 2026-01-07

### Fixed
- Raise clearer error when a well with no data is provided (#37).
- Adjust y-axis behavior for forecast plots (#34).

### Changed
- Bump GitHub Action `actions/checkout` from v5 to v6 (#33).

## [v2.1.0] - 2025-11-18

### Added
- Allow fixing ADCA parameters (`phi`, `p`, `sigma`) across wells (#8).
- Add fallback to Nelder-Mead when BFGS optimization fails (#14).
- Write test-set metrics to `scores.csv` (#15).
- Add snapshot test coverage for ADCA init and run (#13).
- Add ADCA FAQ/docs content and CLI help improvements (#10, #11, #18).

### Fixed
- Remove splitting of pForecast well names (#30).
- Fix interpolation bug triggered on SODIR data (#5).
- Improve error messaging for zero-row PDM response and empty test sets (#12, #16).
- Fix ranking bug (#17).

### Changed
- Bump version to 2.1.0 (#31).
- Improve CLI failure help output and test performance when stdout is not a TTY (#6, #9).
- Refine repository governance and CI setup: Dependabot config, CodeQL workflow, action version bumps, permissions, reviewer setup (#2, #21, #22, #23, #24, #28).

### Documentation
- Clarify references/risk descriptions and update system-card text (#25, #27, #29).
- Remove outdated references to YAML files in docs/readme context (#19).

## [v2.0.0] - 2025-09-17

### Added
- Initial open-source release baseline (`Initial commit for open sourcing code`).

[Unreleased]: https://github.com/equinor/decline-curve-analysis/compare/v2.2.2...HEAD
[v2.2.2]: https://github.com/equinor/decline-curve-analysis/compare/v2.2.1...v2.2.2
[v2.2.1]: https://github.com/equinor/decline-curve-analysis/compare/v2.1.2...v2.2.1
[v2.1.2]: https://github.com/equinor/decline-curve-analysis/compare/v2.1.1...v2.1.2
[v2.1.1]: https://github.com/equinor/decline-curve-analysis/compare/v2.1.0...v2.1.1
[v2.1.0]: https://github.com/equinor/decline-curve-analysis/compare/v2.0.0...v2.1.0
[v2.0.0]: https://github.com/equinor/decline-curve-analysis/releases/tag/v2.0.0
