import React, { useEffect, useRef } from "react";
import SEMBOT_LOGO from "@assets/images/logo.png";
import ButtonWithIcon from "@components/atoms/button/ButtonWithIcon";

type ButtonWithIconProps = React.ComponentProps<typeof ButtonWithIcon>;

export interface SidebarProps {
  components: ButtonWithIconProps[];
  footerComponents: ButtonWithIconProps[];
  isRule: boolean;
  hasMore?: boolean;
  onLoadMore?: () => void;
  isLoading?: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({
  components = [],
  footerComponents = [],
  isRule,
  hasMore = false,
  onLoadMore = () => {},
  isLoading = false,
}) => {
  const observerRef = useRef<IntersectionObserver | null>(null);
  const lastItemRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (observerRef.current) {
      observerRef.current.disconnect();
    }

    observerRef.current = new IntersectionObserver(
      (entries) => {
        const firstEntry = entries[0];
        if (firstEntry.isIntersecting && hasMore && !isLoading) {
          onLoadMore();
        }
      },
      { threshold: 0.1 }
    );

    if (lastItemRef.current) {
      observerRef.current.observe(lastItemRef.current);
    }

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [hasMore, isLoading, onLoadMore]);

  return (
    <div className="h-full">
      <div className="w-full h-full border-r border-indigo-900 bg-semesBlue">
        <div className="flex h-full flex-col">
          {/* Logo Header - Fixed height */}
          <div className="flex-none p-2">
            <div className="w-full flex flex-row mb-2">
              <img src={SEMBOT_LOGO} alt="SEMBOT LOGO" />
              <div className="text-white text-xl mt-2 ml-1">SEMBOT</div>
            </div>
          </div>

          <div className="flex-1 overflow-hidden flex flex-col min-h-0">
            <div className="flex-1 overflow-y-auto px-2 [&::-webkit-scrollbar]:w-0">
              {components.map((buttonProps, index) => (
                <React.Fragment key={index}>
                  <div
                    className="mb-2"
                    onClick={(e) => {
                      e.preventDefault();
                      buttonProps.handleClick?.(e);
                    }}
                  >
                    <ButtonWithIcon
                      {...buttonProps}
                      className={`${buttonProps.styleName} ${
                        buttonProps.isActive ? "bg-white text-semesBlue" : "text-white hover:bg-white/10"
                      }`}
                    />
                  </div>
                  {index === 0 && isRule && <div className="text-white text-xl mb-2">규정 즐겨찾기</div>}
                </React.Fragment>
              ))}

              {/* Loading indicator and sentinel */}
              <div ref={lastItemRef}>{isLoading && <div className="text-white text-center py-2">Loading...</div>}</div>
            </div>
          </div>

          <div className="flex-none px-2 pb-6">
            <hr className="mb-3 border border-gray-100" />
            <div className="space-y-2">
              {footerComponents.map((buttonProps, index) => (
                <div
                  key={index}
                  onClick={(e) => {
                    e.preventDefault();
                    buttonProps.handleClick?.(e);
                  }}
                >
                  <ButtonWithIcon
                    {...buttonProps}
                    className={`${buttonProps.styleName} text-white hover:bg-white/10`}
                  />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
